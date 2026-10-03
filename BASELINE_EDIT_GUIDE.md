# How to edit the research baseline code without changing the scientific protocol

## M1: LFCC-GMM

Use the official ASVspoof baseline implementation. Change only:

- dataset root
- LA protocol paths
- output score path
- random/iteration settings only if required by the implementation

Keep the official LFCC frontend and GMM configuration. Do not replace it with the ZIP's 120-dimensional utterance-pooled LFCC GMM if the goal is an official baseline comparison.

## M2: CQCC-GMM

Use the official ASVspoof baseline implementation. Change only:

- dataset root
- LA protocol paths
- output score path

Keep the CQCC extraction and GMM settings from the reference implementation.

## M3: Log-Mel ResNet-18

The ZIP already contains `LogMelResNet18` and the training engine. The route-1 runner uses:

`raw waveform -> ZIP Log-Mel extractor -> 1-channel 128x126 Log-Mel -> ResNet-18 -> 2-class logits`

This is a controlled baseline against the proposed model because both use the same audio preprocessing and evaluation partition.

## M4: RawNet2

Use the released RawNet2 code if the paper retains the name RawNet2. The ZIP's `LightweightRawNet2` is not the published RawNet2 architecture.

## Proposed model

Keep the ZIP implementation:

`raw waveform -> LFCC 120 -> MLP 32-d embedding`

and

`raw waveform -> Log-Mel 128x126 -> 4-block CNN -> 128-d embedding`

then

`32 + 128 -> 64 -> 2-class output`

The measured total parameter count is 118,294.
