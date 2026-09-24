import os, glob, random, zipfile, tempfile
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import gradio as gr
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# ---- CONFIG: must match your Colab training code ----
MODEL_PATH = "best_crater_cnn.pt"
IMG_SIZE = 128        # size the patches were resized to in training
MEAN, STD = 0.0, 1.0  # 0/1 = plain [0,1] scaling; use 0.5/0.5 if you normalized to [-1,1]
# Output index 0..3 follows your table: 0 = pristine ... 3 = highly degraded (ages are approximate)
CLASSES = ["Pristine / Fresh  ·  < 100 Myr", "Slightly degraded  ·  100–500 Myr",
           "Moderately degraded  ·  0.5–2.0 Gyr", "Highly degraded / Ghost  ·  > 2.0 Gyr"]
SHORT = ["Pristine", "Slight", "Moderate", "Ghost"]
INDEX_TO_ROBBINS = [4, 3, 2, 1]  # class index -> Robbins state; use [1, 2, 3, 4] if you trained on (state - 1)
# ------------------------------------------------------

def block(i, o):
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]

class CraterCNN(nn.Module):  # rebuilt from the saved state_dict keys/shapes
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*block(1, 32), *block(32, 64), *block(64, 128),
                                      nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 4))
    def forward(self, x):
        return self.classifier(self.features(x))

model = CraterCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def prep(img):
    a = np.asarray(img.convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    return torch.from_numpy((a - MEAN) / STD)[None, None]

def probs(imgs):
    with torch.no_grad():
        return F.softmax(model(torch.cat([prep(i) for i in imgs])), 1).numpy()

def cam(img):
    """Grad-CAM on the last conv block, overlaid on the image."""
    img = img.convert("L").resize((256, 256))
    acts = model.features(prep(img))
    logits = model.classifier(acts)
    g, = torch.autograd.grad(logits[0, int(logits.argmax(1))], acts)
    m = F.relu((g.mean((2, 3), keepdim=True) * acts).sum(1, keepdim=True))
    m = F.interpolate(m, size=(256, 256), mode="bilinear")[0, 0].detach().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    base = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    out = (0.55 * base + 0.45 * matplotlib.colormaps["jet"](m)[..., :3]).clip(0, 1)
    return Image.fromarray((out * 255).astype("uint8"))

# ---- Tab 1: single image ----
def classify(img):
    if img is None:
        return None, None
    p = probs([img])[0]
    return {c: float(v) for c, v in zip(CLASSES, p)}, cam(img)

# ---- Tab 2: batch ----
def batch(files):
    if not files:
        return None, None
    rows = []
    for f in files:
        p = probs([Image.open(f)])[0]
        rows.append({"crater_id": os.path.splitext(os.path.basename(f))[0],
                     "predicted_state": INDEX_TO_ROBBINS[int(p.argmax())], "predicted_class": CLASSES[int(p.argmax())],
                     "confidence": round(float(p.max()), 4),
                     **{f"p_{SHORT[i].lower()}": round(float(v), 4) for i, v in enumerate(p)}})
    df = pd.DataFrame(rows)
    path = os.path.join(tempfile.mkdtemp(), "predictions.csv")
    df.to_csv(path, index=False)
    return df, path

# ---- Tab 3: model report (zip with labels.csv + images/<crater_id>.png) ----
def report(zpath, max_n):
    if zpath is None:
        raise gr.Error("Upload a zip containing labels.csv and images/<crater_id>.png")
    d = tempfile.mkdtemp()
    zipfile.ZipFile(zpath).extractall(d)
    files = {os.path.splitext(n)[0]: os.path.join(r, n)
             for r, _, ns in os.walk(d) for n in ns if n.lower().endswith(".png")}
    lab = pd.read_csv(next(glob.iglob(os.path.join(d, "**", "labels.csv"), recursive=True)), dtype={"crater_id": str})
    lab = lab[lab.crater_id.isin(files)].head(int(max_n))
    paths = [files[c] for c in lab.crater_id]
    y_true = np.array([INDEX_TO_ROBBINS.index(v) for v in lab.robbins_state.astype(int)])
    y_pred = np.concatenate([probs([Image.open(p) for p in paths[i:i + 32]]).argmax(1)
                             for i in range(0, len(paths), 32)])
    rep = pd.DataFrame(classification_report(y_true, y_pred, labels=range(4), target_names=CLASSES,
                                             output_dict=True, zero_division=0)).T.round(3)
    rep = rep.reset_index().rename(columns={"index": "class"})
    cm = confusion_matrix(y_true, y_pred, labels=range(4))
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.imshow(cm, cmap="magma")
    ax.set_xticks(range(4), SHORT, rotation=20, fontsize=8); ax.set_yticks(range(4), SHORT, fontsize=8)
    ax.set_xlabel("Predicted class", color="#cbd5ff"); ax.set_ylabel("True class", color="#cbd5ff")
    ax.tick_params(colors="#cbd5ff")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="white" if v < cm.max() / 2 else "black")
    wrong = np.where(y_true != y_pred)[0][:12]
    gallery = [(cam(Image.open(paths[i])), f"true {SHORT[y_true[i]]} -> pred {SHORT[y_pred[i]]}") for i in wrong]
    acc = (y_true == y_pred).mean()
    return f"**Accuracy: {acc:.1%}** on {len(y_true)} patches", rep, fig, gallery

# ---- Tab 4: map ----
def crater_map(lab_file, pred_file, color_by):
    if lab_file is None:
        raise gr.Error("Upload labels.csv")
    df = pd.read_csv(lab_file, dtype={"crater_id": str})
    if pred_file is not None:
        pr = pd.read_csv(pred_file, dtype={"crater_id": str})[["crater_id", "predicted_state"]]
        df = df.merge(pr, on="crater_id")
    if color_by not in df:
        raise gr.Error(f"'{color_by}' not available - upload the predictions CSV from the Batch tab")
    df = df.sample(min(len(df), 30000), random_state=0)
    df[color_by] = df[color_by].astype(str)
    fig = px.scatter(df, x="lon", y="lat", color=color_by, hover_data=["crater_id", "diameter"],
                      render_mode="webgl", category_orders={color_by: sorted(df[color_by].unique())},
                      title="Mars craters (lon/lat)", height=560, template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,10,30,0.55)")
    return fig

# ---------------- Space theme ----------------
def stars(n, seed):
    r = random.Random(seed)
    return ",".join(f"{r.randint(0, 100)}vw {r.randint(-5, 105)}vh {r.choice(['#fff', '#cfd8ff', '#ffd9c2'])}"
                    for _ in range(n))

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
body{background-color:#05060f!important;background-image:radial-gradient(ellipse at 15% 0%,#2b1350 0,transparent 55%),radial-gradient(ellipse at 90% 100%,#5a1d10 0,transparent 50%)!important;background-attachment:fixed!important}
.gradio-container{background:transparent!important;position:relative;z-index:1;max-width:1100px!important}
body::before,body::after{content:"";position:fixed;top:0;left:0;background:transparent;border-radius:50%;pointer-events:none;z-index:0}
body::before{width:1px;height:1px;box-shadow:__S1__;animation:tw 5s ease-in-out infinite alternate}
body::after{width:2px;height:2px;box-shadow:__S2__;animation:tw 3s ease-in-out 1s infinite alternate,fl 70s ease-in-out infinite alternate}
@keyframes tw{from{opacity:.25}to{opacity:1}}
@keyframes fl{to{transform:translateY(-4vh)}}
.shoot{position:fixed;top:-10px;left:var(--x);width:2px;height:90px;background:linear-gradient(#fff0,#fff);opacity:0;pointer-events:none;z-index:0;animation:sh 8s ease-in infinite;animation-delay:var(--d)}
@keyframes sh{0%{transform:translate(0,0) rotate(35deg);opacity:0}4%{opacity:1}22%{transform:translate(-520px,740px) rotate(35deg);opacity:0}100%{opacity:0}}
#hero{background:transparent!important;border:none!important;backdrop-filter:none!important;box-shadow:none!important}
.hero{display:flex;flex-wrap:wrap;gap:28px;align-items:center;justify-content:center;padding:18px 0 6px}
.scene{position:relative;width:210px;height:210px;display:grid;place-items:center}
.planet{position:relative;width:132px;height:132px;border-radius:50%;overflow:hidden;background:radial-gradient(circle at 22% 32%,#e98a55 0 16%,transparent 17%),radial-gradient(circle at 62% 68%,#7d2a08 0 20%,transparent 21%),radial-gradient(circle at 88% 28%,#f2a878 0 11%,transparent 12%),radial-gradient(circle at 45% 12%,#8f3510 0 9%,transparent 10%),#b8410f;background-size:220px 132px;animation:rot 26s linear infinite;box-shadow:0 0 50px rgba(255,90,54,.5)}
.planet::after{content:"";position:absolute;inset:0;border-radius:50%;background:radial-gradient(circle at 30% 28%,transparent 25%,rgba(0,0,0,.75) 100%)}
@keyframes rot{to{background-position:220px 0}}
.orbit{position:absolute;inset:0;border:1px dashed rgba(200,190,255,.25);border-radius:50%;animation:spin 12s linear infinite}
.moon{position:absolute;top:-5px;left:50%;width:10px;height:10px;margin-left:-5px;border-radius:50%;background:#d9d2c5;box-shadow:0 0 10px #fff8}
@keyframes spin{to{transform:rotate(360deg)}}
.title{margin:0;font-size:2.5rem;line-height:1.1;font-weight:700;background:linear-gradient(90deg,#ffc9b3,#ff5a36,#c084fc,#ffc9b3);background-size:300% 100%;-webkit-background-clip:text;background-clip:text;color:transparent;animation:shim 8s linear infinite}
@keyframes shim{to{background-position:300% 0}}
.type{margin-top:10px;font-family:ui-monospace,Menlo,monospace;color:#b9c3ff;white-space:nowrap;overflow:hidden;width:0;max-width:100%;border-right:2px solid #ff8a65;animation:typ 3.6s steps(52) .6s forwards,blk .8s step-end infinite}
@keyframes typ{to{width:52ch}}
@keyframes blk{50%{border-color:transparent}}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
.chips span{padding:3px 11px;border-radius:99px;font-size:.78rem;border:1px solid;background:rgba(255,255,255,.04)}
.chips span:nth-child(1){color:#80cbc4;border-color:#80cbc466}.chips span:nth-child(2){color:#fff176;border-color:#fff17666}.chips span:nth-child(3){color:#ffb74d;border-color:#ffb74d66}.chips span:nth-child(4){color:#ff8a80;border-color:#ff8a8066}
.note{margin-top:10px;font-size:.75rem;color:#8f9ad1}
.gradio-container .block{background:rgba(12,14,42,.55)!important;border:1px solid rgba(160,140,255,.18)!important;border-radius:16px!important;backdrop-filter:blur(10px)}
.tabitem{animation:rise .6s ease both}
@keyframes rise{from{opacity:0;transform:translateY(14px)}}
button[role=tab]{font-weight:600;letter-spacing:.3px}
button[role=tab][aria-selected=true]{color:#ff8a65!important;text-shadow:0 0 14px rgba(255,90,54,.8)}
button.primary{background:linear-gradient(90deg,#ff5a36,#ff2e63,#b44cff,#ff5a36)!important;background-size:300% 100%;color:#fff!important;border:none!important;box-shadow:0 0 22px rgba(255,90,54,.45);transition:transform .2s,box-shadow .2s;animation:shim 6s linear infinite}
button.primary:hover{transform:translateY(-2px) scale(1.02);box-shadow:0 0 36px rgba(255,90,54,.8)}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important}.type{width:52ch}}
""".replace("__S1__", stars(160, 1)).replace("__S2__", stars(70, 2))

HERO = """
<div class="shoot" style="--x:80%;--d:1s"></div><div class="shoot" style="--x:55%;--d:4.5s"></div><div class="shoot" style="--x:96%;--d:7s"></div>
<div class="hero">
  <div class="scene"><div class="orbit"><i class="moon"></i></div><div class="planet"></div></div>
  <div>
    <h1 class="title">Mars Crater Degradation Classifier</h1>
    <div class="type">Reading Mars' erosion history, one crater at a time.</div>
    <div class="chips"><span>Pristine · &lt;100 Myr</span><span>Slight · 100–500 Myr</span><span>Moderate · 0.5–2 Gyr</span><span>Ghost · &gt;2 Gyr</span></div>
    <div class="note">Ages are rough guides per morphology class - the model predicts degradation, not measured age.</div>
  </div>
</div>
"""

V = dict(body_background_fill="transparent", body_text_color="#e8ecff", body_text_color_subdued="#aab4e6",
         block_title_text_color="#c9d2ff", input_background_fill="rgba(6,8,28,.7)",
         border_color_primary="rgba(160,140,255,.2)", button_primary_background_fill="#ff5a36",
         button_primary_text_color="#fff")
THEME = gr.themes.Base(primary_hue="orange", secondary_hue="violet", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Space Grotesk"), "system-ui", "sans-serif"]
                       ).set(**V, **{k + "_dark": v for k, v in V.items()})
STYLE = dict(theme=THEME, css=CSS, js="() => document.body.classList.add('dark')")
NEW_GRADIO = int(gr.__version__.split(".")[0]) >= 6   # Gradio 6 moved theme/css/js to launch()
BLOCKS_KW = {"title": "Mars Crater Degradation Classifier", **({} if NEW_GRADIO else STYLE)}

with gr.Blocks(**BLOCKS_KW) as demo:
    gr.HTML(HERO, elem_id="hero")
    with gr.Tab("🛰️ Classify"):
        with gr.Row():
            inp = gr.Image(type="pil", label="CTX crater patch")
            with gr.Column():
                lbl = gr.Label(num_top_classes=4, label="Predicted class (approx. age)")
                heat = gr.Image(label="Grad-CAM")
        gr.Button("Classify", variant="primary").click(classify, inp, [lbl, heat])
    with gr.Tab("📦 Batch"):
        bf = gr.File(file_count="multiple", file_types=["image"], label="Crater patches (filename = crater_id)")
        bt = gr.Dataframe(label="Predictions")
        bo = gr.File(label="Download predictions.csv")
        gr.Button("Predict all", variant="primary").click(batch, bf, [bt, bo])
    with gr.Tab("📊 Model report"):
        gr.Markdown("Upload a zip with `labels.csv` and `images/<crater_id>.png` (a slice of the dataset, e.g. your validation set).")
        rz = gr.File(file_types=[".zip"], label="Evaluation zip")
        rn = gr.Slider(100, 3000, value=1000, step=100, label="Max patches")
        rs = gr.Markdown()
        rt = gr.Dataframe(label="Per-class precision / recall / F1")
        rp = gr.Plot(label="Confusion matrix")
        rg = gr.Gallery(label="Misclassified (Grad-CAM)", columns=4)
        gr.Button("Evaluate", variant="primary").click(report, [rz, rn], [rs, rt, rp, rg])
    with gr.Tab("🗺️ Crater map"):
        ml = gr.File(file_types=[".csv"], label="labels.csv")
        mp = gr.File(file_types=[".csv"], label="predictions.csv from Batch tab (optional)")
        mc = gr.Dropdown(["robbins_state", "liu_label", "predicted_state"], value="robbins_state", label="Color by")
        mo = gr.Plot()
        gr.Button("Plot", variant="primary").click(crater_map, [ml, mp, mc], mo)

demo.launch(**(STYLE if NEW_GRADIO else {}))
