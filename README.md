# Deep Learning Classification of Martian Impact Crater Degradation from CTX Imagery

A compact convolutional neural network (CNN) that classifies the degradation state of
Martian impact craters directly from Context Camera (CTX) image patches, giving a
reproducible alternative to grading crater preservation by eye.

**Author:** Saksham — Department of Applied Geophysics, IIT (ISM) Dhanbad

---

## Overview

Crater counts are used to estimate the ages of Martian surfaces, and a crater's
preservation state reflects how strongly erosion, infilling, and burial have modified
it. Grading degradation by eye is slow and subjective. This project trains a CNN to
predict the four ordered Robbins & Hynek degradation states (pristine → highly
degraded) from CTX image patches, and ships the model as an interactive web
application for single-image, batch, evaluation, and map-based use.

## Dataset

- **Source:** [Mars Crater CTX Images with Dual Degradation Labels](https://doi.org/10.5281/zenodo.20801959) (Hammer, 2026)
- **Size:** 33,901 grayscale 1024×1024 patches
- **Base imagery:** [Global CTX Mosaic of Mars](https://doi.org/10.1029/2024EA003555) (Dickson et al., 2024)
- **Labels:** Four ordered Robbins & Hynek preservation states (state 1 = most
  degraded, state 4 = most pristine)
- **Preprocessing:** Patches converted to grayscale, downsampled to a small fixed
  size, split into training and held-out validation sets (5,086 validation patches)

## Model

A compact CNN (~0.42M parameters):

- 4 convolutional layers (32 → 256 filters) with batch normalization and ReLU
- 3 max-pooling stages
- Global average pooling
- 2-layer classifier head

**Training:** PyTorch, 10 epochs; the checkpoint with the best validation accuracy
(epoch 8) is retained.

**Evaluation:** overall accuracy, per-class precision/recall/F1, confusion matrix, and
Grad-CAM to visualize which image regions drive each prediction.

## Results

| Metric | Value |
|---|---|
| Overall validation accuracy | 61% |
| Macro-F1 | 0.58 |
| Majority-class baseline accuracy | 39% |
| Predictions within 1 state of the true label | 98% |

- Best per-class F1: state 1, most degraded (0.74; precision 0.87, recall 0.65)
- Worst per-class F1: state 2 (0.41)
- States 3 and 4: F1 of 0.66 and 0.52 respectively
- 96% of misclassifications (1,882 of 1,970) fall between *adjacent* states —
  consistent with degradation being a continuum graded into discrete classes, with
  state 2 as the main source of ambiguity

See `figures/` (or the abstract PDF) for the training curves, per-class metrics and
confusion matrix, and an example Grad-CAM overlay.

## Web Application

The trained model is deployed as an interactive web app with four modes:

- **Single-image** — classify one CTX patch and view its Grad-CAM overlay
- **Batch** — classify a folder/set of patches at once
- **Evaluation** — compute accuracy, per-class metrics, and confusion matrix on a
  labeled set
- **Map** — view predicted degradation states in a spatial/map context

> Fill in the specific run command / URL / entry point for your deployment, e.g.:
> ```bash
> python app.py
> # or
> streamlit run app.py
> ```

## Repository Structure

> Adjust this to match your actual layout — placeholder below:

```
.
├── data/               # dataset download / preprocessing scripts
├── models/             # model definition and saved checkpoints
├── train.py            # training loop
├── evaluate.py         # metrics, confusion matrix, Grad-CAM
├── app.py              # web application (single/batch/evaluation/map modes)
├── figures/            # training log, per-class metrics, Grad-CAM examples
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

## Training

```bash
python train.py --epochs 10 --data-dir data/
```

## Future Work

- Ordinal-aware loss functions
- Class weighting for intermediate (harder-to-separate) states
- Transfer learning from larger pretrained networks
- Evaluation on an independent, spatially separated test set
- Comparison against the Liu et al. (2024) degradation labels
- Combining predicted degradation with crater size-frequency analysis for relative
  age estimation; extension to the Moon and other bodies

## References

1. Hartmann, W. K. & Neukum, G. (2001). Cratering chronology and the evolution of
   Mars. *Space Science Reviews*, 96, 165–194.
2. Palafox, L. F. et al. (2017). Automated detection of geological landforms on Mars
   using convolutional neural networks. *Computers & Geosciences*, 101, 48–56.
3. Lee, C. (2019). Automated crater detection on Mars using deep learning.
   *Planetary and Space Science*, 170, 16–28.
4. Robbins, S. J. & Hynek, B. M. (2012). A new global database of Mars impact
   craters ≥1 km. *Journal of Geophysical Research: Planets*, 117.
5. Liu, D. et al. (2024). A global catalog of Martian impact craters with actual
   boundaries and degradation states. *International Journal of Applied Earth
   Observation and Geoinformation*.
6. Hammer, A. (2026). *Mars Crater CTX Images with Dual Degradation Labels* (v1.0.0)
   [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20801959
7. Dickson, J. L. et al. (2024). The Global CTX Mosaic of Mars. *Earth and Space
   Science*, 11(7). https://doi.org/10.1029/2024EA003555
8. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks
   via gradient-based localization. *Proceedings of the IEEE International
   Conference on Computer Vision (ICCV)*, 618–626.
