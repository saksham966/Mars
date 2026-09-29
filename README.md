---
title: Mars Crater Degradation
emoji: 🪐
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
---

# 🪐 Mars Crater Degradation Classifier

A CNN that looks at a grayscale CTX (Context Camera) image of a Martian impact crater and predicts how degraded it is — from pristine to a barely visible "ghost" crater. Built on top of the **Mars Crater CTX Images with Dual Degradation Labels** dataset ([Zenodo record 20801959](https://zenodo.org/records/20801959)).

Try it: upload a crater patch on the **Classify** tab, or explore the other tabs below.

## What it predicts

The model outputs one of four Robbins & Hynek preservation states. The age ranges are rough guides from crater morphology studies, not a measured age — the model predicts degradation, not age.

| State | Class | Approx. age |
|---|---|---|
| 1 | Highly degraded / Ghost | > 2.0 Gyr |
| 2 | Moderately degraded | 0.5–2.0 Gyr |
| 3 | Slightly degraded | 100–500 Myr |
| 4 | Pristine / Fresh | < 100 Myr |

## Tabs

- **🛰️ Classify** — upload one crater patch, see the predicted class with a confidence bar per class, plus a Grad-CAM overlay showing which part of the image drove the prediction.
- **📦 Batch** — upload many patches at once (filename = crater ID) and download all predictions as a CSV.
- **📊 Model report** — upload a small zip (`labels.csv` + `images/<crater_id>.png`) from a validation slice of the dataset. Returns accuracy, a per-class precision/recall/F1 table, a confusion matrix, and a gallery of misclassified craters with Grad-CAM.
- **🗺️ Crater map** — plot craters by longitude/latitude from `labels.csv`, optionally colored by the predictions CSV from the Batch tab.

## Model

A compact CNN (~0.42M parameters): four conv blocks (32→256 filters) with batch norm and ReLU, max-pooling, global average pooling, and a small dense classifier head. Trained in PyTorch on Google Colab.

On a 5,086-patch held-out validation set: **61% accuracy**, macro-F1 0.58 (vs. a 39% majority-class baseline). Errors are concentrated between neighboring degradation states — 98% of predictions land within one state of the true label. The most degraded and most pristine classes are easiest to separate; the two intermediate states are the main source of confusion.

## Running locally

```bash
pip install -r requirements.txt gradio
python app.py
```

Open the printed `http://127.0.0.1:7860` link. `best_crater_cnn.pt` must be in the same folder as `app.py`.

## Notes / known limitations

- The dataset's `liu_label` (a separate 3-class scale) is not used here — this model was trained on the 4-class `robbins_state` labels only.
- Grad-CAM can pick up on image annotations (scale bars, IDs, north arrows) baked into some CTX figure exports rather than the crater itself — worth a visual sanity-check on results.
- Validation accuracy, not an untouched test set — treat it as an estimate.

## Credits

- Dataset: Hammer, A. (2026). *Mars Crater CTX Images with Dual Degradation Labels* (v1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20801959
- Base imagery: Dickson, J. L., Ehlmann, B. L., Kerber, L., & Fassett, C. I. (2024). The Global CTX Mosaic of Mars. *Earth and Space Science*.
- Degradation states: Robbins, S. J., & Hynek, B. M. (2012). A new global database of Mars impact craters ≥1 km. *JGR: Planets*.
- Built with [Gradio](https://gradio.app).
