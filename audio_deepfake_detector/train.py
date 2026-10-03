"""
Training Engine for Audio Deepfake Detection.
Implements class-weighted cross-entropy training with:
  - AdamW optimizer + Cosine Annealing LR scheduler
  - Epoch-level logging of all metrics
  - Best-model checkpoint saving
  - Separate train loops for Neural models (DualStream, ResNet, RawNet2)
    and sklearn GMM baselines
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .evaluate import compute_all_metrics, format_metrics_table


def get_class_weights(labels_list, num_classes=2, device='cpu'):
    """
    Computes inverse-frequency class weights to counteract label imbalance.
    ASVspoof 2019 LA train: ~90% spoof, ~10% bonafide.

    Args:
        labels_list (list): All training labels (0 or 1)
        num_classes (int): Number of classes
        device (str/torch.device): Target device

    Returns:
        torch.Tensor: Class weights tensor
    """
    counts = np.bincount(labels_list, minlength=num_classes).astype(float)
    total  = counts.sum()
    # Inverse frequency: minority class gets higher weight
    weights = total / (num_classes * counts)
    weights = weights / weights.sum() * num_classes  # normalize to sum = num_classes
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch_neural(model, loader, feature_extractor, optimizer, criterion, device, scaler=None):
    """
    Trains a neural model (DualStream, ResNet-18, or standalone CNN/MLP) for one epoch.

    Args:
        model             : PyTorch nn.Module (DualStreamDetector, ResNet-18, etc.)
        loader            : DataLoader yielding {waveform, label, attack_id, ...}
        feature_extractor : DualStreamFeatureExtractor or None (for RawNet2 raw input)
        optimizer         : torch.optim optimizer
        criterion         : Loss function (CrossEntropyLoss with class weights)
        device            : torch.device
        scaler            : Optional torch.cuda.amp.GradScaler for mixed precision

    Returns:
        float: Average training loss for this epoch
    """
    model.train()
    if feature_extractor:
        feature_extractor.eval()

    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        wav    = batch['waveform'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        if feature_extractor is not None:
            # Dual-stream or CNN input: extract LFCC + Log-Mel
            with torch.no_grad():
                lfcc, mel = feature_extractor(wav)
            lfcc = lfcc.to(device)
            mel  = mel.to(device)

            if scaler:
                with torch.amp.autocast(device_type=device.type if hasattr(device, 'type') else 'cpu'):
                    logits = model(lfcc, mel)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(lfcc, mel)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        else:
            # RawNet2-style: raw waveform input
            if scaler:
                with torch.amp.autocast(device_type=device.type if hasattr(device, 'type') else 'cpu'):
                    logits = model(wav)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(wav)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_neural(model, loader, feature_extractor, device, is_dual_stream=True, threshold=None):
    """
    Evaluates a neural model over a data loader.

    Args:
        model             : Trained model
        loader            : DataLoader yielding batches
        feature_extractor : DualStreamFeatureExtractor or None
        device            : Target device
        is_dual_stream    : Whether model uses dual stream
        threshold         : Optional fixed decision threshold (e.g. frozen DEV threshold tau*)

    Returns:
        dict: Full metrics + raw labels/scores/attack_ids for per-attack breakdown
    """
    model.eval()
    if feature_extractor:
        feature_extractor.eval()

    all_labels     = []
    all_scores     = []
    all_attack_ids = []

    for batch in loader:
        wav        = batch['waveform'].to(device)
        labels     = batch['label'].cpu().numpy()
        attack_ids = batch['attack_id']

        if feature_extractor is not None:
            lfcc, mel = feature_extractor(wav)
            lfcc = lfcc.to(device)
            mel  = mel.to(device)
            logits = model(lfcc, mel)
        else:
            logits = model(wav)

        probs  = F.softmax(logits, dim=-1)[:, 1]  # P(Y=1|x) = spoof prob
        scores = probs.cpu().numpy()

        all_labels.extend(labels.tolist())
        all_scores.extend(scores.tolist())
        all_attack_ids.extend(attack_ids)

    metric_threshold = 0.5 if threshold is None else threshold
    metrics = compute_all_metrics(all_labels, all_scores, threshold=metric_threshold)
    metrics['labels']     = np.array(all_labels)
    metrics['scores']     = np.array(all_scores)
    metrics['attack_ids'] = np.array(all_attack_ids)
    if threshold is not None:
        metrics['eval_threshold_used'] = float(threshold)
    return metrics


def train_neural_model(
    model,
    train_loader,
    val_loader,
    feature_extractor,
    device,
    epochs=30,
    lr=1e-4,
    weight_decay=1e-4,
    class_weights=None,
    save_dir='checkpoints',
    model_name='model',
    is_dual_stream=True
):
    """
    Full training loop for neural models.

    Args:
        model             : nn.Module to train
        train_loader      : DataLoader for training set
        val_loader        : DataLoader for validation/dev set
        feature_extractor : DualStreamFeatureExtractor (or None for end-to-end)
        device            : torch.device
        epochs            : Number of training epochs
        lr                : Initial learning rate
        weight_decay      : L2 regularization weight
        class_weights     : Optional torch.Tensor of class weights
        save_dir          : Directory to save best checkpoint
        model_name        : Name tag for checkpoint file
        is_dual_stream    : Whether model takes (lfcc, mel) or raw wav

    Returns:
        dict: Training history {train_loss, val_acc, val_f1, val_auc, val_eer per epoch}
    """
    os.makedirs(save_dir, exist_ok=True)

    # Loss function with class weighting
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

    # AdamW optimizer (as per paper)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # Cosine Annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        'train_loss': [], 'val_acc': [], 'val_f1': [],
        'val_auc': [], 'val_eer': [], 'val_precision': [], 'val_recall': []
    }

    best_eer   = float('inf')
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}  |  epochs={epochs}  |  lr={lr}")
    print(f"  Device: {device}  |  Class weights: {class_weights}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Training step
        train_loss = train_one_epoch_neural(
            model, train_loader, feature_extractor, optimizer, criterion, device
        )

        # Validation step
        val_metrics = evaluate_neural(model, val_loader, feature_extractor, device, is_dual_stream)

        scheduler.step()
        elapsed = time.time() - t0

        # Record history
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['roc_auc'])
        history['val_eer'].append(val_metrics['eer'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])

        print(
            f"  Epoch {epoch:>3}/{epochs}"
            f"  Loss={train_loss:.4f}"
            f"  Val Acc={val_metrics['accuracy']*100:.2f}%"
            f"  F1={val_metrics['f1']*100:.2f}%"
            f"  AUC={val_metrics['roc_auc']*100:.2f}%"
            f"  EER={val_metrics['eer']*100:.2f}%"
            f"  [{elapsed:.1f}s]"
        )

        # Save best checkpoint (lowest EER)
        if val_metrics['eer'] < best_eer:
            best_eer   = val_metrics['eer']
            best_epoch = epoch
            best_thresh = val_metrics.get('eer_threshold', 0.5)
            ckpt_path  = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save({
                'epoch':         epoch,
                'model_state':   model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'best_eer':      best_eer,
                'eer_threshold': best_thresh,
                'val_metrics':   val_metrics
            }, ckpt_path)

    print(f"\n  Best EER={best_eer*100:.2f}% at epoch {best_epoch}")
    print(f"  Checkpoint saved: {os.path.join(save_dir, model_name + '_best.pth')}")

    history['best_eer']   = best_eer
    history['best_epoch'] = best_epoch
    return history


def train_gmm_baseline(
    gmm_classifier,
    feature_extractor_fn,
    train_samples,
    verbose=True
):
    """
    Trains a GMM baseline model (M1 LFCC-GMM or M2 CQCC-GMM).

    Args:
        gmm_classifier     : GMMClassifier instance
        feature_extractor_fn: Callable (waveform tensor) -> numpy feature array
        train_samples      : List of dicts {waveform_path, label}
        verbose            : Print progress

    Returns:
        GMMClassifier: Fitted classifier
    """
    from .dataset import load_audio_file

    X_bona, X_spoof = [], []

    for i, sample in enumerate(train_samples):
        if verbose and (i % 500 == 0):
            print(f"  Extracting features [{i}/{len(train_samples)}]...", flush=True)

        wav = load_audio_file(sample['path'])
        feat = feature_extractor_fn(wav.unsqueeze(0))

        if isinstance(feat, torch.Tensor):
            feat = feat.squeeze(0).numpy()

        if sample['label'] == 0:
            X_bona.append(feat)
        else:
            X_spoof.append(feat)

    X_bona  = np.vstack(X_bona)
    X_spoof = np.vstack(X_spoof)

    gmm_classifier.fit(X_bona, X_spoof, verbose=verbose)
    return gmm_classifier


def evaluate_gmm_baseline(gmm_classifier, feature_extractor_fn, eval_samples, verbose=True, threshold=None):
    """
    Evaluates a fitted GMM baseline on evaluation samples.

    Args:
        gmm_classifier      : GMMClassifier instance
        feature_extractor_fn: Callable
        eval_samples        : List of sample dicts
        verbose             : Print progress
        threshold           : Optional fixed decision threshold (e.g. from DEV)

    Returns:
        dict: Full metrics dict with labels, scores, attack_ids
    """
    from .dataset import load_audio_file

    all_labels, all_scores, all_attack_ids = [], [], []

    for i, sample in enumerate(eval_samples):
        if verbose and (i % 2000 == 0):
            print(f"  Evaluating GMM [{i}/{len(eval_samples)}]...", flush=True)

        wav  = load_audio_file(sample['path'])
        feat = feature_extractor_fn(wav.unsqueeze(0))

        if isinstance(feat, torch.Tensor):
            feat = feat.squeeze(0).numpy()

        score = float(gmm_classifier.score(feat.reshape(1, -1))[0])
        all_labels.append(sample['label'])
        all_scores.append(score)
        all_attack_ids.append(sample.get('attack_id', '-'))

    metric_threshold = 0.5 if threshold is None else threshold
    metrics = compute_all_metrics(all_labels, all_scores, threshold=metric_threshold)
    metrics['labels']     = np.array(all_labels)
    metrics['scores']     = np.array(all_scores)
    metrics['attack_ids'] = np.array(all_attack_ids)
    if threshold is not None:
        metrics['eval_threshold_used'] = float(threshold)
    return metrics
