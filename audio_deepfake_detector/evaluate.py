"""
Standardized Evaluation Metrics for ASVspoof Anti-Spoofing.
Implements all metrics from the paper (Section 4, Equations 4-10):
  - Accuracy, Precision, Recall, F1
  - Equal Error Rate (EER) via bisection on ROC curve
  - ROC-AUC
  - Per-attack breakdown (A07-A19)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)


def compute_eer(labels, scores):
    """
    Computes the Equal Error Rate (EER) from binary labels and detection scores.
    EER = threshold tau* where P_fa(tau*) = P_miss(tau*)

    Uses exact linear interpolation at the ROC curve crossing point.

    Args:
        labels (array-like): Ground truth binary labels (0=bonafide, 1=spoof)
        scores (array-like): Detection scores (higher => more likely spoof)

    Returns:
        float: EER value in range [0, 1]
        float: EER threshold tau*
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    diff = fpr - fnr

    # Check for exact crossing
    exact = np.where(np.isclose(diff, 0.0))[0]
    if len(exact) > 0:
        i = int(exact[0])
        return float(fpr[i]), float(thresholds[i])

    # Check for sign change and linearly interpolate
    crossing = np.where(diff[:-1] * diff[1:] < 0)[0]
    if len(crossing) > 0:
        i = int(crossing[0])
        d0, d1 = diff[i], diff[i + 1]
        alpha = -d0 / (d1 - d0)
        eer = fpr[i] + alpha * (fpr[i + 1] - fpr[i])
        threshold = thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i])
        return float(eer), float(threshold)

    # Fallback to nearest point if no clean crossing exists
    i = int(np.nanargmin(np.abs(diff)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i])


def compute_all_metrics(labels, scores, threshold=0.5, pos_label=1):
    """
    Computes the complete ASVspoof metric set from paper Table 3.

    Args:
        labels    (array-like): Ground truth binary labels
        scores    (array-like): P(Y=1|x) spoof detection scores
        threshold (float):      Decision threshold (default: 0.5)
        pos_label (int):        Positive (spoof) class label (default: 1)

    Returns:
        dict: {
            'accuracy', 'precision', 'recall', 'f1',
            'roc_auc', 'eer', 'eer_threshold',
            'pfa', 'pmiss',
            'tp', 'tn', 'fp', 'fn'
        }
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    preds  = (scores >= threshold).astype(int)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float('nan')

    eer, eer_thresh = compute_eer(labels, scores)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # P_fa = FP / (FP + TN),  P_miss = FN / (TP + FN)
    pfa   = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    pmiss = fn / (tp + fn) if (tp + fn) > 0 else float('nan')

    return {
        'accuracy':      float(acc),
        'precision':     float(prec),
        'recall':        float(rec),
        'f1':            float(f1),
        'roc_auc':       float(auc),
        'eer':           float(eer),
        'eer_threshold': float(eer_thresh),
        'pfa':           float(pfa),
        'pmiss':         float(pmiss),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def compute_per_attack_metrics(labels, scores, attack_ids, threshold=0.5):
    """
    Computes per-attack EER and accuracy breakdown for seen/unseen attacks (A07-A19).

    Args:
        labels     (array-like): Ground truth labels per sample
        scores     (array-like): Detection scores per sample
        attack_ids (array-like): Attack ID per sample (e.g. 'A07', '-' for bonafide)
        threshold  (float):      Decision threshold

    Returns:
        dict: {attack_id: {'eer', 'accuracy', 'n_samples'}}
    """
    labels     = np.asarray(labels)
    scores     = np.asarray(scores)
    attack_ids = np.asarray(attack_ids)

    # Overall bonafide samples (attack_id == '-')
    unique_attacks = sorted(set(a for a in attack_ids if a != '-'))
    results = {}

    for atk in unique_attacks:
        # Pair each attack against all bonafide samples
        atk_mask  = (attack_ids == atk)
        bona_mask = (attack_ids == '-')
        mask = atk_mask | bona_mask

        if mask.sum() < 2:
            continue

        atk_labels = labels[mask]
        atk_scores = scores[mask]
        preds      = (atk_scores >= threshold).astype(int)

        acc = accuracy_score(atk_labels, preds)
        try:
            eer, _ = compute_eer(atk_labels, atk_scores)
        except Exception:
            eer = float('nan')

        results[atk] = {
            'eer':       float(eer),
            'accuracy':  float(acc),
            'n_samples': int(mask.sum())
        }

    return results


def format_metrics_table(metrics_dict, title="Evaluation Results"):
    """
    Formats metric results as a pretty ASCII table for console output.
    """
    lines = [
        "=" * 60,
        f"  {title}",
        "=" * 60,
        f"  Accuracy  : {metrics_dict['accuracy']*100:7.2f}%",
        f"  Precision : {metrics_dict['precision']*100:7.2f}%",
        f"  Recall    : {metrics_dict['recall']*100:7.2f}%",
        f"  F1 Score  : {metrics_dict['f1']*100:7.2f}%",
        f"  ROC-AUC   : {metrics_dict['roc_auc']*100:7.2f}%",
        f"  EER       : {metrics_dict['eer']*100:7.2f}%  (tau*={metrics_dict['eer_threshold']:.4f})",
        "-" * 60,
        f"  P_fa      : {metrics_dict['pfa']*100:7.2f}%",
        f"  P_miss    : {metrics_dict['pmiss']*100:7.2f}%",
        "-" * 60,
        f"  TP={metrics_dict['tp']:6d}  FP={metrics_dict['fp']:6d}",
        f"  FN={metrics_dict['fn']:6d}  TN={metrics_dict['tn']:6d}",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_per_attack_table(per_attack_dict, title="Per-Attack Breakdown"):
    """
    Formats per-attack EER results as a table.
    """
    lines = [
        "=" * 50,
        f"  {title}",
        "=" * 50,
        f"  {'Attack':<10} {'EER (%)':>10} {'Acc (%)':>10} {'N':>8}",
        "-" * 50,
    ]
    for atk in sorted(per_attack_dict.keys()):
        r = per_attack_dict[atk]
        lines.append(
            f"  {atk:<10} {r['eer']*100:>10.2f} {r['accuracy']*100:>10.2f} {r['n_samples']:>8}"
        )
    lines.append("=" * 50)
    return "\n".join(lines)
