# Claims that must be updated after the experiments

Do not retain the following as fixed facts from the current PDF:

- Proposed Accuracy = 95.82%
- Proposed ROC-AUC = 98.72%
- Proposed EER = 4.18%
- Grad-CAM masking confidence drop = 38.4%
- stable performance at 30/20 dB and breakdown at 0 dB

These must be replaced by the measured outputs.

The evaluation attacks are A07-A19. A16 and A19 are evaluation attacks associated with algorithms seen in training/development, while the evaluation set as a whole contains A07-A19. Avoid describing every A07-A19 attack as completely unseen.

Use neutral wording such as:

"The evaluation partition contains attacks A07-A19, including attacks that reuse or vary algorithms represented in the training/development conditions."

For XAI, use:

"Attribution validation by perturbation masking"

rather than implying that Grad-CAM alone proves causality.
