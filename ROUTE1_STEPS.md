# Route 1: Reproducible ASVspoof 2019 LA Experiment

## What is being used from the existing ZIP

Keep the existing feature/model code for the proposed method:

- `audio_deepfake_detector/features.py`
- `audio_deepfake_detector/dataset.py`
- `audio_deepfake_detector/models/lfcc_mlp.py`
- `audio_deepfake_detector/models/cnn_stream.py`
- `audio_deepfake_detector/models/dual_stream.py`
- `audio_deepfake_detector/train.py`
- `audio_deepfake_detector/kaggle_data.py`

The supplied repository already passes the Phase 1-5 architecture/training-engine tests. The proposed network has 118,294 total parameters.

## Important changes for the paper

1. Do not use the old `checkpoints/results_table3.json` numbers.
2. Do not use the current in-repository GMM implementation for the final M1/M2 paper numbers; it pools features differently from the official ASVspoof GMM baselines.
3. Use the official ASVspoof LFCC-GMM and CQCC-GMM implementations for M1/M2.
4. Use the official RawNet2 implementation for M4 if you keep the name `RawNet2`. The existing `LightweightRawNet2` is only RawNet2-inspired.
5. M3 is the controlled Log-Mel ResNet-18 implementation already in the ZIP. Describe it as your reproduced ResNet-18 baseline unless you exactly reproduce another paper's configuration.
6. The supplied `evaluate.py` uses linear interpolation at the ROC crossing instead of using only the nearest ROC point.
7. XAI and robustness are run after the proposed checkpoint is trained.

## Colab setup

```bash
!pip install -q kagglehub soundfile scipy scikit-learn matplotlib pandas
```

Upload/copy this project into Colab and make the project root the current directory.

## Step 1: verify the code before touching the real dataset

```bash
!python audio_deepfake_detector/test_pipeline.py
!python audio_deepfake_detector/test_phase45.py
!python audio_deepfake_detector/test_phase67.py
```

All three test suites should pass cleanly.

## Step 2: download and verify ASVspoof 2019 LA

```bash
!python -c "from audio_deepfake_detector.kaggle_data import download_asvspoof2019, get_la_paths, quick_dataset_summary; p=download_asvspoof2019(); q=get_la_paths(p); quick_dataset_summary(q)"
```

Do not proceed to final results unless TRAIN, DEV and EVAL are all found.

Expected LA sizes are approximately:
- Train: 2,580 bona fide + 22,800 spoof
- Dev: 2,548 bona fide + 22,296 spoof
- Eval: 7,355 bona fide + 63,882 spoof

## Step 3: smoke-test the proposed model

Run only a few epochs first:

```bash
!python run_route1.py --kaggle --models dual_stream --epochs 2 --batch_size 32 --num_workers 2 --save_dir smoke_results
```

Check that a checkpoint is produced and the evaluation metrics are finite.

## Step 4: final proposed model + M3 baseline

For the final paper run:

```bash
!python run_route1.py --kaggle --models dual_stream resnet18 --epochs 20 --batch_size 32 --num_workers 2 --save_dir results_route1
```

If the GPU run is stable and time permits, repeat with 30 epochs. Do not change the reported epoch count without recording it in the paper.

## Step 5: run XAI

Use the trained proposed checkpoint:

```bash
!python run_xai.py --kaggle --checkpoint results_route1/proposed_dual_stream_best.pth --max_samples 100 --top_k_bands 30 --output_dir results_route1/xai
```

The important result is the measured mean/median relative confidence drop. Do not write `38.4%` unless the run actually produces it.

## Step 6: run robustness

First smoke test:

```bash
!python run_robustness.py --kaggle --checkpoint results_route1/proposed_dual_stream_best.pth --max_samples 1000 --output results_route1/robustness_1000.json
```

For the paper, use the full evaluation partition if runtime permits:

```bash
!python run_robustness.py --kaggle --checkpoint results_route1/proposed_dual_stream_best.pth --max_samples 100000 --output results_route1/robustness_full.json
```

The script evaluates Clean, 30 dB, 20 dB, 10 dB and 0 dB.

## Step 7: M1/M2 official baselines

Use the official ASVspoof baseline implementations. The ASVspoof 2019 site explicitly provides LFCC-GMM and CQCC-GMM baseline software, and the ASVspoof 2021 baseline repository provides Python versions of both baselines.

Official source:
- https://www.asvspoof.org/index2019.html
- https://github.com/asvspoof-challenge/2021

Configure the baseline code to use the ASVspoof 2019 LA train/dev/eval protocol and audio directories. Save the evaluation score files.

Do not copy published accuracy/AUC/EER values into your table. Generate the scores with the actual run.

## Step 8: M4 RawNet2

If the paper says `RawNet2`, use the released RawNet2 implementation rather than the current lightweight substitute:

https://github.com/eurecom-asp/rawnet2-antispoofing

Use the logical-access 2019 setup and save the evaluation scores. If you instead keep the ZIP's lightweight model, rename the paper row to `Lightweight Raw-Waveform Baseline (RawNet2-inspired)`.

## Step 9: final comparison table

The final Table 3 must contain only runs produced by the same ASVspoof 2019 LA evaluation protocol:

M1 LFCC-GMM
M2 CQCC-GMM
M3 Log-Mel ResNet-18
M4 RawNet2
Proposed Dual-Stream

Report Accuracy, Precision, Recall, F1, ROC-AUC and EER only when they have been computed from the corresponding saved score files.

## Step 10: update the paper

Replace every current claimed result with the actual output. In particular, verify:

- Table 3 values
- per-attack A07-A19 results
- XAI confidence drop
- high-frequency attribution ratio
- Clean/30/20/10/0 dB results
- parameter count
- training epochs and class-weight method

The current paper's numerical claims are placeholders until these runs are completed.
