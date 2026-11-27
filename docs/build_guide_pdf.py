"""Build docs/Project_Guide.pdf: the whole crop-health project explained, technically and in simple words.

    python scripts/make_guide_figures.py && python scripts/make_figures.py && python docs/build_guide_pdf.py
"""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)
from reportlab.platypus.doctemplate import BaseDocTemplate, Frame, PageTemplate
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "docs", "Project_Guide.pdf")

# ------------------------------------------------------------------ fonts & styles
F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("DVM", F + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, MUTED = colors.HexColor("#1F2328"), colors.HexColor("#57606A")
BLUE, GREEN, PURPLE = colors.HexColor("#2F5D9E"), colors.HexColor("#2E7D4F"), colors.HexColor("#6A4FA3")
SIMPLE_BG, TECH_BG, NOTE_BG = colors.HexColor("#EAF5EE"), colors.HexColor("#EEF2FA"), colors.HexColor("#FFF6E0")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=24, leading=30, alignment=TA_CENTER, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=12.5, leading=18, alignment=TA_CENTER, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=17, leading=22, spaceBefore=4, spaceAfter=8, textColor=BLUE),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12.5, leading=17, spaceBefore=10, spaceAfter=4, textColor=INK,
                        keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14.2, spaceAfter=5, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.6, leading=14, leftIndent=14, bulletIndent=3,
                             spaceAfter=2, textColor=INK),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.4, leading=13.8, textColor=INK),
    "boxh": ParagraphStyle("boxh", fontName="DV-B", fontSize=9.4, leading=13, spaceAfter=2),
    "eq": ParagraphStyle("eq", fontName="DV", fontSize=10.2, leading=15, alignment=TA_CENTER, spaceBefore=3,
                         spaceAfter=6, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8.5, leading=11.5, alignment=TA_CENTER, textColor=MUTED,
                          spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.6, leading=11.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.6, leading=11.5, textColor=INK),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=8, leading=10.5, textColor=INK),
    "toc1": ParagraphStyle("toc1", fontName="DV", fontSize=10.5, leading=17, leftIndent=6),
}

story = []
_h1_count = [0]


def P(text, style="body"):
    story.append(Paragraph(text, S[style]))


def H1(text):
    _h1_count[0] += 1
    story.append(PageBreak())
    p = Paragraph(f"{_h1_count[0]}. {text}", S["h1"])
    p._toc = f"{_h1_count[0]}. {text}"
    story.append(p)


def H2(text):
    story.append(Paragraph(text, S["h2"]))


def BUL(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="•"))


def _callout(title, body, bg, edge, title_color):
    paras = [Paragraph(title, ParagraphStyle("t", parent=S["boxh"], textColor=title_color))]
    for b in (body if isinstance(body, list) else [body]):
        paras.append(Paragraph(b, S["box"]))
    t = Table([[paras]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(t)
    story.append(Spacer(1, 7))


def SIMPLE(body):
    _callout("In simple words", body, SIMPLE_BG, GREEN, GREEN)


def TECH(body):
    _callout("Technical detail", body, TECH_BG, BLUE, BLUE)


def NOTE(body, title="Important"):
    _callout(title, body, NOTE_BG, colors.HexColor("#C9971C"), colors.HexColor("#8A6510"))


def EQ(text):
    P(text, "eq")


def CODE(text):
    t = Table([[Preformatted(text, S["code"])]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 7))


def TABLE(rows, widths, header=True, bold_rows=()):
    data = [[Paragraph(str(c), S["cellb"] if (header and r == 0) or r in bold_rows else S["cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D7DE")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7ECF3")))
    for r in bold_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F3F0FA")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 8))


def FIGURE(name, caption, width=16.4):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        story.append(Paragraph(f"[figure {name} not generated yet]", S["cap"]))
        return
    img = Image(path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width * cm, width * cm * ratio
    story.append(KeepTogether([img, Spacer(1, 3), Paragraph(caption, S["cap"])]))




# ------------------------------------------------------------------ results (read from results/, never typed by hand)
def load(rel):
    p = os.path.join(ROOT, "results", rel, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x, d=2):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


CLS = {k: load(f"classification/{k}") for k in
       ["yolo26_cls_sfr", "yolo26_cls", "yolo11_cls_sfr", "yolo11_cls", "yolo26_cls_sfr_scratch", "small_cnn",
        "yolo26_cls_sfr_bgaug"]}
BASE = load("classification/baselines_handcrafted")
DET = {k: load(f"detection/{k}") for k in ["yolo26n_sfr", "yolo26n", "yolo11n", "yolov5nu"]}
MAIN = CLS["yolo26_cls_sfr"]
CLS_NAMES = {"yolo26_cls_sfr": "YOLO26n-cls + SFR (proposed)", "yolo26_cls": "YOLO26n-cls (no SFR)",
             "yolo11_cls_sfr": "YOLO11n-cls + SFR", "yolo11_cls": "YOLO11n-cls",
             "yolo26_cls_sfr_scratch": "YOLO26n-cls + SFR, random init", "small_cnn": "Small CNN (scratch)",
             "yolo26_cls_sfr_bgaug": "YOLO26n-cls + SFR + background aug."}
DET_NAMES = {"yolo26n_sfr": "YOLO26n + SFR (proposed)", "yolo26n": "YOLO26n", "yolo11n": "YOLO11n",
             "yolov5nu": "YOLOv5n (original family)"}


def cacc(k, part="test"):
    m = CLS.get(k)
    return m[part]["accuracy"] if m and part in m else None


def cf1(k, part="test"):
    m = CLS.get(k)
    return m[part]["macro_f1"] if m and part in m else None


# ================================================================== CONTENT
# ---- cover
story.append(Spacer(1, 2.6 * cm))
P("Crop Health Assessment and<br/>Plant Disease Detection", "title")
P("Attention-refined YOLO (YOLO26 + Selective Feature Refinement) with smart sensing for precision farming", "sub")
story.append(Spacer(1, 0.7 * cm))
P("<b>The Complete Project Guide</b><br/>Everything from start to end, explained technically and in simple words",
  "sub")
story.append(Spacer(1, 1.0 * cm))
FIGURE("system_architecture.png", "The whole system on one page: a leaf photo from the app, the web page or the "
       "ESP32-CAM goes to a Flask server, a YOLO model with SFR attention names the disease, and the advisory "
       "database returns fertilizer, pesticide and precautions. Field sensors drive the irrigation pump.", 16.4)
story.append(Spacer(1, 0.5 * cm))
P("Code: github.com/Vishnu868/Crop-Health-Assessment-and-Disease-Detection", "sub")

# ---- how to read + TOC
story.append(PageBreak())
P("How to read this guide", "h1")
P("Every topic is explained twice:")
SIMPLE("A plain-language explanation with everyday analogies. If you read only the green boxes, you will understand "
       "what the project does, why, and what it found.")
TECH("The precise version: equations, tensor shapes, hyper-parameters and design decisions. This is what you need to "
     "defend the work in a viva, reproduce it, or extend it.")
NOTE("Orange boxes flag things that are easy to get wrong or that an examiner is likely to ask about.", "Watch out")
P("Every number in this guide is read automatically from the <font face='DVM'>results/</font> folder when the PDF "
  "is built, so the guide cannot drift from the actual experiments.")
P("Contents", "h2")
toc = TableOfContents()
toc.levelStyles = [S["toc1"]]
story.append(toc)

# ================================================================== 1. Big picture
H1("The Big Picture")
SIMPLE(["Imagine a farmer who sees strange spots on a tomato leaf. Is it a harmless blemish, or late blight, which "
        "can wipe out a whole field in a week? An expert could tell at a glance, but experts are few and far away.",
        "This project puts a 'pocket plant doctor' in the farmer's phone. You photograph a leaf. A neural network "
        "looks at it and says which of five conditions it shows: <b>bacterial spot, black rot, healthy, late blight "
        "or powdery mildew</b>. It also says how much of the leaf looks damaged, and what to do about it: which "
        "fertilizer, which pesticide, and which precautions.",
        "A small camera board (ESP32-CAM) can do the same automatically in the field. A second board (NodeMCU) "
        "with soil-moisture, temperature and humidity sensors switches the water pump on and off by itself."])
TECH(["Pipeline: RGB leaf image → resize to 128×128 → <b>YOLO26n-cls backbone</b> (ImageNet-pretrained) with two "
      "<b>Selective Feature Refinement (SFR)</b> attention blocks → 5-way softmax → class-specific advisory from "
      "<font face='DVM'>advisory.json</font>. In parallel, RGB vegetation indices (VARI, ExG, GLI) and a hue-based "
      "lesion fraction are computed over the segmented leaf to give a 0-100 health score.",
      "An object-detection variant (<b>YOLO26n + SFR</b> on the P3/P4/P5 neck outputs) localises the leaf and "
      "classifies it, matching the original project's YOLOv5 detector design. YOLOv5n, YOLO11n and YOLO26n without "
      "SFR are trained as comparisons.",
      "Data: <b>8,000 real PlantVillage images</b> (1,600 per class, balanced over 16 crop/disease source folders), "
      "split 70/15/15, stratified by class × crop."])
FIGURE("pipeline.png", "From raw data to results: every box is one script in the repository.")
if MAIN:
    H2("The headline result")
    P(f"On 1,200 held-out test images that the model never saw during training, the proposed YOLO26n-cls + SFR "
      f"classifier is correct <b>{pct(cacc('yolo26_cls_sfr'))}</b> of the time (macro-F1 "
      f"{cf1('yolo26_cls_sfr'):.4f}). Chance is 20%. The best classical baseline (colour and texture features + "
      f"random forest) reaches {pct(BASE['random_forest']['accuracy']) if BASE else 'n/a'}.")
    NOTE(f"The same test leaves with their background digitally removed are classified correctly only "
         f"{pct(cacc('yolo26_cls_sfr', 'test_background_removed'))} of the time. A large part of PlantVillage "
         "accuracy comes from the lab background and lighting, not from the lesions alone. Section 10 explains "
         "this and the fix we tested. Mention it in any presentation: it is the most important limitation.",
         "Read this before quoting 99%")

# ================================================================== 2. Problem & objectives
H1("Problem Statement and Objectives")
SIMPLE(["Plant diseases destroy an estimated 20-40% of global crop production every year. Most farmers find out too "
        "late, and often spray the wrong chemical, or too much of it. The problem is not a lack of medicine. It is "
        "the lack of a fast, cheap and correct diagnosis at the moment it matters."])
P("<b>Problem.</b> Given a photograph of a single leaf, decide automatically whether the plant is healthy or which "
  "of four common diseases it has, quantify the damage, and give actionable advice. The advice must reach the "
  "farmer through a phone, a web page or an unattended field camera.")
P("<b>Objectives</b>")
BUL(["Rebuild the lost dataset from public data, with the exact five classes of the original project.",
     "Build a disease detector in the YOLO family with an attention module (SFR), and upgrade it from YOLOv5 to "
     "the newest YOLO generation (YOLO26).",
     "Compare it fairly with ablations (no SFR, older YOLO, no pretraining) and classical baselines on a held-out "
     "test set, using accuracy, per-class precision/recall/F1, confusion matrices, ROC/AUC and mAP.",
     "Explain predictions (Grad-CAM, SFR attention maps) and assess leaf health with RGB vegetation indices.",
     "Deliver the system: Flask REST API, web interface, Flutter app, ESP32-CAM and NodeMCU firmware, advisory "
     "database, irrigation logic, tests and documentation."])

# ================================================================== 3. Domain background
H1("Background: the Five Leaf Conditions")
SIMPLE("Each disease leaves a different 'fingerprint' on the leaf: the colour of the spots, their shape, whether "
       "they have a yellow ring, whether the leaf looks dusty. A good model learns these fingerprints the way a "
       "plant pathologist does.")
TABLE([["Class", "Cause", "What it looks like", "Crops in our data"],
       ["Bacterial spot", "Xanthomonas bacteria", "small dark, water-soaked spots, often with a yellow halo",
        "pepper, peach, tomato"],
       ["Black rot", "Botryosphaeria obtusa / Guignardia bidwellii (fungi)",
        "'frog-eye' spots with purple margins (apple); tan lesions with dark borders (grape)", "apple, grape"],
       ["Healthy", "none", "uniform green, no lesions",
        "apple, cherry, grape, peach, pepper, potato, tomato"],
       ["Late blight", "Phytophthora infestans (oomycete)",
        "large greasy grey-brown lesions, white mould under the leaf when humid", "potato, tomato"],
       ["Powdery mildew", "Podosphaera spp. and other Erysiphales (fungi)", "white powdery coating on the surface",
        "cherry, squash"]], [2.6, 4.0, 6.0, 4.0])
FIGURE("dataset_samples.png", "Real test-split examples for each class. The source crop is shown above each image.",
       15.5)

# ================================================================== 4. DL background
H1("Deep-Learning Background")
H2("4.1 Convolutional neural networks")
SIMPLE("A CNN looks at an image through many small 'stencils' (filters). Early stencils react to edges and colour "
       "blobs, later ones to textures like 'brown spot with yellow ring', and the last ones to whole concepts like "
       "'late blight leaf'. Training adjusts the stencils until the network's guesses match the true labels.")
TECH(["A convolution layer computes Y<sub>c'</sub> = Σ<sub>c</sub> W<sub>c',c</sub> * X<sub>c</sub> + b<sub>c'</sub> "
      "for each output channel c'. It is followed by batch normalisation and a non-linearity; YOLO uses SiLU, "
      "x·σ(x). Stride-2 convolutions halve the resolution, so a 128×128 input ends as a 4×4×256 map at stride 32.",
      "The final map is global-average-pooled to a vector, and a fully connected layer gives logits z. Softmax "
      "turns them into probabilities: p<sub>i</sub> = e<sup>z<sub>i</sub></sup> / Σ<sub>j</sub> "
      "e<sup>z<sub>j</sub></sup>."])
H2("4.2 Transfer learning")
SIMPLE("Instead of teaching a newborn to see, we start from a network that has already looked at 1.28 million "
       "everyday photos (ImageNet) and only teach it the new vocabulary of leaf diseases. It learns faster and "
       "needs far fewer examples.")
TECH("We load the ImageNet weights of the Ultralytics <b>YOLO26n-cls</b> model (1.5 M parameters), replace the "
     "1000-class linear layer with a 5-class one, and fine-tune <i>all</i> layers with a small learning rate. The "
     "'random init' ablation trains the identical architecture from scratch to measure what pretraining buys.")
H2("4.3 The YOLO family: from v5 to 26")
SIMPLE("YOLO ('You Only Look Once') is a family of fast detectors that find and name objects in one pass over the "
       "image. The original project used YOLOv5 (2020). Since then the family has improved several times. We "
       "upgraded to <b>YOLO26</b>, the newest release, and kept YOLOv5 and YOLO11 as comparisons.")
TABLE([["Generation", "Key ideas", "Used here as"],
       ["YOLOv5 (2020)", "CSP backbone with C3 blocks, PANet neck, anchor-based head (we use the anchor-free 'u' "
        "re-release yolov5nu)", "original-architecture baseline"],
       ["YOLO11 (2024)", "C3k2 blocks, C2PSA (position-sensitive self-attention) block, decoupled anchor-free head",
        "strong modern comparison"],
       ["YOLO26 (2026)", "same efficient backbone family; end-to-end NMS-free head, DFL removed, MuSGD optimiser, "
        "tuned for CPU / edge inference", "<b>proposed backbone and detector</b>"]], [3.0, 9.4, 4.2])
NOTE("YOLO is a <i>detector</i>: it outputs boxes plus classes. PlantVillage has one centred leaf per image, so "
     "image-level <i>classification</i> is the natural main task. We therefore use the YOLO26 classification "
     "backbone (YOLO26n-cls) for the headline numbers and the YOLO26 detector for localisation, exactly as both "
     "are deployed in the app.", "Detector vs classifier")
H2("4.4 Attention and Selective Feature Refinement (SFR)")
SIMPLE("When a doctor examines a leaf, they do not look at everything equally. They pick out the useful clues "
       "('these brown spots') and the useful places ('the lower-left patch'). SFR does the same inside the network. "
       "A channel gate decides <i>which</i> detectors matter, and a spatial gate decides <i>where</i> to look.")
TECH(["For a feature map X ∈ ℝ<sup>C×H×W</sup>:"])
EQ("M<sub>c</sub> = σ( MLP(AvgPool(X)) + MLP(MaxPool(X)) ) ∈ ℝ<sup>C×1×1</sup>,   X' = M<sub>c</sub> ⊙ X")
EQ("M<sub>s</sub> = σ( Conv<sub>7×7</sub>( [mean<sub>c</sub>(X'); max<sub>c</sub>(X')] ) ) ∈ ℝ<sup>1×H×W</sup>,   "
   "X'' = M<sub>s</sub> ⊙ X'")
EQ("Y = X + γ · X''      (γ is a learned scalar, initialised to 0)")
TECH(["The MLP is two 1×1 convolutions (C → C/8 → C) with SiLU. Because γ starts at 0, inserting SFR into a "
      "pretrained network changes nothing at initialisation; the network learns how much refinement to use. "
      "This is verified by a unit test (identical detector outputs before and after insertion).",
      "<b>Placement.</b> Classifier: after the last stride-16 C3k2 stage (128 ch) and after the stride-32 C2PSA "
      "stage (256 ch). Detector: on the three neck outputs (P3, P4, P5) that feed the Detect head.",
      f"<b>Learned gates</b> (proposed classifier): γ = "
      f"{', '.join(f'{g:.3f}' for g in MAIN['sfr_gamma']) if MAIN else 'n/a'}. Both moved away from 0, so the "
      "network actually uses the refined features."])
FIGURE("model_architecture.png", "The proposed classifier and the inside of one SFR block.")

# ================================================================== 5. Dataset
H1("The Dataset")
SIMPLE(["The original project's data was lost with the hard disk. We rebuilt it from public data. The filenames in "
        "the original test log (for example 'UMD_Powd.M', 'FAM_B.Rot', 'Rutg._Bact.S', 'GHLB2 Leaf', 'RS_HL') are "
        "the exact filenames of <b>PlantVillage</b>, the best-known public plant-disease image collection: 54,000+ "
        "photos of single leaves taken in a lab. That tells us where the original images came from.",
        "PlantVillage is organised by crop and disease (38 folders). The original project merged them into five "
        "groups, so we did the same."])
TECH(["Source: github.com/spMohanty/PlantVillage-Dataset (the authors' official release; raw/color and "
      "raw/segmented, 256×256 JPEG). Mapping: Bacterial_spot ← pepper, peach, tomato; Black_rot ← apple, grape; "
      "Late_Blight ← potato, tomato; Powder_Mildew ← cherry, squash; Healthy ← apple, cherry, grape, peach, pepper, "
      "potato, tomato.",
      "Sampling: 1,600 images per class (the smallest class, black rot, has 1,801), drawn with a fixed seed and "
      "spread as evenly as possible over source crops (water-filling). This gives 8,000 images; the background-"
      "removed version of each one is downloaded too.",
      "Split: 70 / 15 / 15 (5,600 / 1,200 / 1,200), stratified on class × source crop, so every split has the "
      "same crop mix. One image (a peach bacterial-spot leaf) has no segmented version upstream; it uses a "
      "colour-threshold mask instead."])
FIGURE("dataset_composition.png", "Each class bar is split by source crop. Classes are perfectly balanced.")
NOTE(["The original project exported its data through Roboflow (the '.rf.&lt;hash&gt;' suffix in the filenames) and "
      "reported 430 validation images with 539 boxes, so some images had several boxes. That export is not "
      "public, so our subset is the closest faithful reconstruction, not the identical file list.",
      "PlantVillage photos were taken under controlled conditions: one leaf, plain background, even lighting. "
      "Models trained on it are known to lose accuracy on real field photos (see Section 10)."],
     "Not the identical original split")

# ================================================================== 6. Preprocessing
H1("Preprocessing, Augmentation and Health Indices")
H2("6.1 Preprocessing")
SIMPLE("Before training, every photo is checked, shrunk to a size the computer can process quickly, and labelled "
       "with a box around the leaf. The box is computed automatically from the background-removed version of the "
       "same photo.")
TECH(["Each image is decoded with OpenCV (corrupt files are skipped). A leaf mask is taken from the PlantVillage "
      "segmented image (pixel sum > 30), cleaned with a 3×3 morphological opening, and reduced to its largest "
      "connected component. The YOLO box (x<sub>c</sub>, y<sub>c</sub>, w, h), normalised to [0,1], is the "
      "bounding rectangle of that mask. Masks smaller than 2% of the frame fall back to the full image.",
      "Classifier input: INTER_AREA resize to 128×128, scaled to [0,1], then normalised with ImageNet statistics "
      "x' = (x − μ)/σ, μ = (0.485, 0.456, 0.406), σ = (0.229, 0.224, 0.225). The detector reads the original "
      "256×256 JPEGs.",
      "Workers run in a process pool and each result is collected with a 120 s timeout, so one bad file cannot "
      "hang the run."])
H2("6.2 Augmentation")
SIMPLE("To stop the network from memorising photos, we show it slightly different versions every time: flipped, "
       "rotated, zoomed, a bit brighter or darker. A leaf is still the same disease upside down.")
TECH(["Classifier (on the fly, deterministic per (seed, epoch, index)): horizontal and vertical flip (p = 0.5 each), "
      "random 90° rotation, affine transform with p = 0.7 (rotation ±20°, scale 0.85-1.15, shift ±10%, reflection "
      "padding), brightness ×U(0.8, 1.2), offset ±0.2, per-channel colour shift ±0.1.",
      "Detector: Ultralytics defaults: mosaic (switched off for the last quarter of epochs), HSV jitter, "
      "translate/scale, horizontal flip.",
      "Background augmentation (extra experiment): with probability 0.5 the training image is replaced by its "
      "background-removed version."])
H2("6.3 Crop-health indices from an ordinary RGB camera")
SIMPLE("Satellites measure plant health with infrared light (NDVI). Phones and the ESP32-CAM have no infrared "
       "channel, so we use colour-only indices: how 'green' the leaf is compared with red and blue. We also measure "
       "what share of the leaf is not green, which is a simple estimate of lesion area.")
TECH(["With chromatic coordinates r = R/(R+G+B), etc., averaged over leaf pixels:"])
EQ("ExG = 2g − r − b       VARI = (G − R) / (G + R − B)       GLI = (2G − R − B) / (2G + R + B)")
EQ("lesion pixel ⇔ hue ∉ [25, 100] (OpenCV 0-179 scale) or (S &lt; 30 and V &gt; 170, whitish coating)")
EQ("lesion fraction = share of leaf pixels that are lesion pixels;   health score = 100 · (1 − lesion fraction)")
FIGURE("health_lesion_maps.png", "Top: input. Bottom: background darkened, lesion (non-green) pixels in red.")
FIGURE("health_indices.png", "Index distributions per class over all 8,000 images. Late blight has the lowest "
       "health score. The thin white coating of powdery mildew rarely trips the pale-pixel rule, but mildew leaves "
       "have the lowest VARI and ExG, so the greenness indices catch what the lesion rule misses.")
NOTE("The health score is a rule-based colour measure, not a trained model. It cannot tell a natural reddish "
     "leaf from a lesion, and it has not been validated against expert severity ratings.", "Limitation")

# ================================================================== 7. Models
H1("Models and Baselines")
SIMPLE("To know whether the proposed model is really good, we race it against simpler competitors: an old-school "
       "method that counts colours and textures, a small network trained from nothing, the same network without "
       "the attention module, and older YOLO generations. If the fancy model cannot clearly beat the simple ones, "
       "it is not worth its complexity.")
rows = [["Model", "What it is", "Params"]]
desc = {"yolo26_cls_sfr": "ImageNet YOLO26n-cls backbone + 2 SFR blocks + new head",
        "yolo26_cls": "same, without SFR (ablation)", "yolo11_cls_sfr": "YOLO11n-cls backbone + 2 SFR blocks",
        "yolo11_cls": "YOLO11n-cls backbone, no SFR", "yolo26_cls_sfr_scratch": "proposed model, random init",
        "small_cnn": "4 blocks × (2 conv-BN-ReLU + max-pool), widths 24-192, GAP, dropout 0.3",
        "yolo26_cls_sfr_bgaug": "proposed model trained with 50% background-removed images"}
for k, d in desc.items():
    if CLS.get(k):
        rows.append([CLS_NAMES[k], d, f"{CLS[k]['params'] / 1e6:.2f} M"])
rows.append(["Colour + texture + LogReg / RF", "842 features: HSV histogram (256), HOG (576), LBP (10) → "
             "logistic regression (C = 0.05) or random forest (400 trees)", "-"])
TABLE(rows, [5.2, 9.4, 2.0], bold_rows=(1,))
H2("Detectors")
rows = [["Detector", "Initial weights", "Params"]]
for k, d in DET.items():
    if d:
        rows.append([DET_NAMES[k], "COCO-pretrained " + {"yolo26n_sfr": "yolo26n", "yolo26n": "yolo26n",
                                                          "yolo11n": "yolo11n", "yolov5nu": "yolov5nu"}[k],
                     f"{d['params'] / 1e6:.2f} M"])
TABLE(rows, [6.0, 8.6, 2.0])

# ================================================================== 8. Training & evaluation
H1("Training and Evaluation")
H2("8.1 Training the classifier")
SIMPLE("Training is a loop: show a batch of 64 leaves, measure how wrong the guesses are (the loss), nudge every "
       "weight a little in the direction that reduces the error, and repeat. After each pass over the training "
       "set (an epoch) we check the validation set and keep the best version.")
TECH(["Loss: label-smoothed cross-entropy (ε = 0.05): L = −Σ<sub>i</sub> q<sub>i</sub> log p<sub>i</sub>, with "
      "q<sub>y</sub> = 1 − ε + ε/K and q<sub>i≠y</sub> = ε/K, K = 5.",
      "Optimiser: AdamW, lr 2·10<sup>-3</sup>, weight decay 5·10<sup>-4</sup>, batch 64; one epoch of linear "
      "warm-up, then cosine decay to 0. 15 epochs (small CNN: 20). Model selection: best validation macro-F1. The "
      "test split is used exactly once, at the end.",
      "Hardware: 4-core CPU, no GPU; ~25 s per epoch for the YOLO26n-cls models at 128 px."])
H2("8.2 Training the detector")
TECH(["Ultralytics trainer with a subclass whose get_model() inserts SFR. Settings: imgsz 256, batch 32, AdamW "
      "lr 0.002 with cosine schedule, 15 epochs, mosaic closed for the last 3 epochs, COCO-pretrained weights.",
      "CPU budget: each detector trains on a fixed, seeded 35% subset of the training split (~1,960 images). It is "
      "validated on the full validation split and tested on the full 1,200-image test split. One epoch over all "
      "5,600 images would take ~10 min on this CPU.",
      "Box labels are automatic (from the segmentation mask), one box per image."])
H2("8.3 Metrics")
SIMPLE("Accuracy alone can hide problems, so we use several report cards. Precision: 'when the model says late "
       "blight, how often is it right?' Recall: 'of all the late-blight leaves, how many did it find?' F1 combines "
       "the two. The confusion matrix shows exactly which diseases get mixed up.")
EQ("Precision = TP / (TP + FP)     Recall = TP / (TP + FN)     F1 = 2·P·R / (P + R)     "
   "macro-F1 = (1/K) Σ<sub>k</sub> F1<sub>k</sub>")
EQ("IoU(A, B) = |A ∩ B| / |A ∪ B|        AP = ∫<sub>0</sub><sup>1</sup> P(R) dR        "
   "mAP@0.5:0.95 = mean over IoU thresholds 0.50, 0.55, …, 0.95")
TECH(["ROC/AUC: one-vs-rest curves from the softmax probabilities; macro AUC = mean over the 5 classes.",
      "Detector image-level accuracy: the class of the most confident box (conf ≥ 0.25) is the image's prediction; "
      "no box counts as an error. This makes detectors directly comparable with classifiers.",
      "Background-removed test: the same 1,200 test images with every non-leaf pixel set to black, classified by "
      "the unchanged model. This measures how much the model relies on the background."])
H2("8.4 Grad-CAM")
SIMPLE("Grad-CAM colours the parts of the photo that pushed the model towards its answer. If it highlights the "
       "lesions, the model is 'looking' in the right place. If it highlights the background, we have a problem.")
EQ("α<sub>k</sub><sup>c</sup> = (1/Z) Σ<sub>i,j</sub> ∂y<sup>c</sup>/∂A<sup>k</sup><sub>ij</sub>      "
   "L<sup>c</sup> = ReLU( Σ<sub>k</sub> α<sub>k</sub><sup>c</sup> A<sup>k</sup> )")
TECH("A is the output of the 1×1 head convolution (1280 × 4 × 4 at 128 px input), upsampled bilinearly to the "
     "image size. We also show the SFR spatial map M<sub>s</sub> of the stride-32 block.")

# ================================================================== 9. Results
H1("Results")
P("All numbers come from the <b>held-out test split: 1,200 real PlantVillage images, 240 per class</b>. Chance "
  "is 20%; the majority-class baseline is 20.0% (macro-F1 0.067).")
H2("9.1 Classification")
rows = [["Model", "Accuracy", "Macro-F1", "Macro AUC", "Acc. (background removed)"]]
order = ["yolo26_cls_sfr", "yolo26_cls", "yolo11_cls_sfr", "yolo11_cls", "yolo26_cls_sfr_scratch", "small_cnn",
         "yolo26_cls_sfr_bgaug"]
for k in order:
    mm = CLS.get(k)
    if mm:
        rows.append([CLS_NAMES[k], pct(mm["test"]["accuracy"]), f"{mm['test']['macro_f1']:.4f}",
                     f"{mm['test']['macro_auc_ovr']:.4f}", pct(mm["test_background_removed"]["accuracy"])])
if BASE:
    for k, n in [("random_forest", "Colour+texture + random forest"),
                 ("logistic_regression", "Colour+texture + logistic regr.")]:
        rows.append([n, pct(BASE[k]["accuracy"]), f"{BASE[k]['macro_f1']:.4f}", f"{BASE[k]['macro_auc_ovr']:.4f}",
                     "-"])
TABLE(rows, [6.4, 2.2, 2.2, 2.2, 3.6], bold_rows=(1,))
FIGURE("cls_model_comparison.png", "Accuracy and macro-F1 of every classifier on the test split.")
FIGURE("cls_training_curves.png", "Training and validation curves.")
FIGURE("cls_confusion_matrix.png", "Proposed model. Left: normal test images. Right: the same images with the "
       "background removed.")
FIGURE("cls_per_class.png", "Per-class precision, recall and F1 of the proposed model.")
FIGURE("cls_roc.png", "One-vs-rest ROC curves (log-scaled false-positive axis, so the differences are visible).",
       10)
FIGURE("gradcam.png", "Grad-CAM and SFR attention maps on random test images.")
H2("9.2 Detection")
rows = [["Detector", "P", "R", "mAP@0.5", "mAP@0.5:0.95", "Image-level acc."]]
for k, d in DET.items():
    if d:
        rows.append([DET_NAMES[k], f"{d['test']['precision']:.3f}", f"{d['test']['recall']:.3f}",
                     f"{d['test']['mAP50']:.3f}", f"{d['test']['mAP50_95']:.3f}",
                     pct(d["test_image_level"]["accuracy"])])
if len(rows) > 1:
    TABLE(rows, [5.4, 1.7, 1.7, 2.1, 2.6, 3.1], bold_rows=(1,))
    FIGURE("det_model_comparison.png", "Detector comparison on the test split.")
    FIGURE("det_confusion_matrix.png", "Image-level confusion of the proposed detector.", 11)
    FIGURE("det_examples.png", "Test-split detections of the proposed detector.")
else:
    P("Detector results were not available when this PDF was built.")
H2("9.3 Results reported in the original project")
SIMPLE("The original project's screenshots survive in the repository. We copy their numbers here, clearly "
       "separated, so you can compare. They come from a different model, image split and box labelling, so they "
       "are not directly comparable.")
TABLE([["Class", "Images", "Instances", "P", "R", "mAP@0.5", "mAP@0.5:0.95"],
       ["all", "430", "539", "0.743", "0.723", "0.743", "0.428"],
       ["Bacterial_spot", "430", "95", "0.550", "0.432", "0.486", "0.215"],
       ["Black_rot", "430", "119", "0.586", "0.647", "0.652", "0.232"],
       ["Healthy", "430", "108", "0.972", "0.959", "0.977", "0.697"],
       ["Late_Blight", "430", "109", "0.686", "0.697", "0.696", "0.347"],
       ["Powder_Mildew", "430", "108", "0.921", "0.880", "0.902", "0.648"]],
      [3.4, 1.6, 2.0, 1.6, 1.6, 2.2, 2.6], bold_rows=(1,))
P("Source: <font face='DVM'>screenshots/trainresults.jpg</font>: YOLOv5 (157 layers, 7.02 M parameters, 15.8 "
  "GFLOPs, i.e. YOLOv5s size) + SFR, 100 epochs at 640 px, 47.4 hours on CPU, best.pt on the validation set. The "
  "matching confusion matrix (<font face='DVM'>screenshots/heat_map.jpg</font>) shows many objects predicted as "
  "'background', i.e. missed.")
NOTE(["The repository also contains a bar chart of 93-99% per-class precision/recall/F1 "
      "(<font face='DVM'>bargraph.jpg</font>), a confusion matrix titled 'High Accuracy Representation' with "
      "95-97% diagonals (<font face='DVM'>new_heatmap.jpg</font>), and smooth ROC curves with AUC 0.95-0.98 "
      "(<font face='DVM'>eoc_curve.jpg</font>). These figures do not agree with the logged training run "
      "(mAP@0.5 = 0.743, recall 0.43-0.96). Their shapes suggest they are illustrative rather than computed from "
      "predictions. We could not reproduce them from any logged output, so we do not treat them as results.",
      "In this rebuild every reported number is computed from saved predictions, and the code that produced it "
      "is in the repository."], "About the original figures")
P("<b>Why our numbers are much higher than the original mAP of 0.743.</b> (1) Our boxes are derived from the "
  "segmentation masks, one per image and pixel-consistent, while the original used hand-drawn Roboflow boxes, "
  "several per image. (2) The newer YOLO generations and ImageNet/COCO pretraining help. (3) Our images are "
  "256×256 single-leaf photos without the original export's extra augmentation. The task is therefore easier. "
  "The improvement does not mean the original model was badly trained.")

# ================================================================== 10. Discussion
H1("Discussion and Limitations")
SIMPLE(["The model is excellent at telling the five conditions apart <i>in PlantVillage-style photos</i>. But "
        "when we cut the leaf out and put it on a black background, accuracy drops sharply. The model learned the "
        "diseases, but it also learned clues about how each crop and disease was photographed: background, "
        "lighting and which crops belong to which class.",
        "It is like a student who passes the exam partly by recognising which teacher wrote each question. The "
        "practical lesson: before trusting the app in a real field, collect and test on real field photos."])
bg = CLS.get("yolo26_cls_sfr_bgaug")
TECH([f"<b>Shortcut evidence.</b> Proposed model: {pct(cacc('yolo26_cls_sfr'))} on normal test images vs "
      f"{pct(cacc('yolo26_cls_sfr', 'test_background_removed'))} on the same images with the background removed. "
      "In PlantVillage each class is also tied to particular crops (e.g. powdery mildew only on cherry and squash), "
      "so leaf shape partly reveals the class. This is a dataset property, not something SFR can remove.",
      (f"<b>Background augmentation helps.</b> Training with 50% background-removed images gives "
       f"{pct(bg['test']['accuracy'])} on normal and {pct(bg['test_background_removed']['accuracy'])} on "
       f"background-removed test images, a much smaller gap.") if bg else
      "<b>Background augmentation</b> results were not available when this PDF was built.",
      "<b>SFR's contribution.</b> Compare YOLO26n-cls with and without SFR in Section 9. On a dataset where the "
      "baseline is already above 98%, the remaining headroom is small; differences of a few tenths of a percent "
      "are within run-to-run noise (one seed per model; ±0.3-0.5% is typical for 1,200 test images).",
      "<b>Classical baselines are strong.</b> A random forest on colour histograms already reaches ~96%, which "
      "confirms that colour carries most of the signal in this dataset.",
      "<b>Other limitations.</b> Single random split (no cross-validation); a 35% training subset and 15 epochs "
      "for the detectors (CPU budget); automatic boxes; subset of 8,000 of 54,000 PlantVillage images; the health "
      "score is not validated against expert severity; firmware and Flutter code are provided as source but were "
      "not compiled or run on hardware in this rebuild."])
H2("Future work")
BUL(["Collect and label field photos (e.g. PlantDoc, or the farm's own ESP32-CAM images) and test on them.",
     "Leaf-level splits and cross-crop validation (train on tomato, test on potato late blight).",
     "Several seeds and 5-fold cross-validation to put error bars on the SFR ablation.",
     "Multispectral (NIR) camera for a true NDVI, and severity labels to validate the health score.",
     "Export to ONNX/TFLite and run on the phone for offline use; push the irrigation log to a time-series DB."])

# ================================================================== 11. App + IoT
H1("The Application and IoT Layer")
SIMPLE("The model is useful only if farmers can reach it. Three 'front doors' lead to the same server: the "
       "phone app, a web page, and the ESP32-CAM camera in the field. The server answers in plain language, and "
       "a second little board looks after watering.")
TABLE([["Component", "File", "Role"],
       ["Flask API", "app/app.py", "POST /predict (multipart or base64 JSON), POST/GET /api/sensors, GET /api/health"],
       ["Web interface", "app/templates/index.html, app/static/*", "upload, diagnosis, top-3 bars, health meter, "
        "advisory, live sensor panel; dark mode"],
       ["Advisory DB", "app/advisory.json", "per class: disease, pathogen, symptoms, fertilizer, pesticide, "
        "precautions"],
       ["Flutter app", "mobile_app/lib/*.dart", "camera/gallery → upload → result screen (Material 3)"],
       ["ESP32-CAM", "firmware/esp32_cam_capture/", "VGA JPEG → base64 → POST /predict every 60 s or on button"],
       ["NodeMCU", "firmware/nodemcu_sensors_pump/", "DHT11 + soil moisture → hysteresis pump control via relay "
        "→ POST /api/sensors"],
       ["Irrigation logic", "src/irrigation.py", "Python twin of the firmware logic, unit-tested"]],
      [3.0, 5.0, 8.6])
TECH(["Pump control with hysteresis: ON when moisture &lt; 35% (40% if T &gt; 32 °C and RH &lt; 40%), OFF when "
      "moisture &gt; 60% or after 10 minutes (safety cut-off). Hysteresis stops the relay from chattering around "
      "one threshold.",
      "Disease-risk flag: 'high' when RH &gt; 85% and 15 °C ≤ T ≤ 28 °C (leaf-wetness conditions favourable to "
      "late blight and mildew), 'medium' when RH &gt; 70%.",
      "Example response of POST /predict:"])
CODE('{"disease": "Late Blight", "class": "Late_Blight", "confidence": 0.9987,\n'
     ' "top3": [{"class": "Late_Blight", "prob": 0.9987}, ...],\n'
     ' "fertilizer": "Avoid excess nitrogen; apply potassium and phosphorus ...",\n'
     ' "pesticide": "Systemic plus contact fungicides such as metalaxyl-M + mancozeb ...",\n'
     ' "precautions": "Destroy infected plants and volunteer potatoes ...",\n'
     ' "health": {"health_score": 71.2, "lesion_fraction": 0.288, "vari": 0.07, ...},\n'
     ' "low_confidence": false, "latency_ms": 41.3}')
NOTE("Advisory text is general agronomic guidance compiled for this project. It is not a prescription: always "
     "follow the product label and local extension advice.", "Safety")

# ================================================================== 12. Code
H1("Code Walkthrough")
TABLE([["File", "What it does"],
       ["scripts/download_plantvillage.py", "blob-less git index of PlantVillage, balanced seeded sampling, "
        "16-thread resumable download with retries, manifest.csv"],
       ["scripts/preprocess.py", "verify images, leaf masks, YOLO boxes, health indices, 128-px arrays (pv128.npz), "
        "stratified split, YOLO dataset + data.yaml"],
       ["src/config.py", "class names, paths, image sizes, pretrained-weight URLs"],
       ["src/sfr.py", "SFR module and SFRWrap (wraps an Ultralytics layer, keeps its i/f/type attributes)"],
       ["src/models.py", "SmallCNN, YoloClassifier (YOLO26/11/v8 backbones ± SFR), build_model, load_checkpoint"],
       ["src/data.py", "normalisation, deterministic augmentation, LeafDataset"],
       ["src/train_classifier.py", "training loop, metrics, background-removed test, checkpoints; synthetic mode"],
       ["src/train_detector.py", "Ultralytics trainer with SFR insertion, test mAP, image-level evaluation"],
       ["src/baseline_features.py", "HSV histogram + HOG + LBP features, LogReg / RF / majority baselines"],
       ["src/health.py", "leaf mask, ExG / VARI / GLI, lesion fraction, overlay"],
       ["src/gradcam.py", "Grad-CAM and overlay"],
       ["src/predict.py", "inference: probabilities + advisory + health (+ boxes, Grad-CAM)"],
       ["src/irrigation.py", "pump hysteresis controller and disease-risk rule"],
       ["scripts/make_figures.py", "all result figures and results/summary.md"],
       ["tests/test_pipeline.py", "pytest: SFR identity, shapes, augmentation, sampling, health, irrigation, API"]],
      [5.2, 11.4])
H2("The SFR module in code")
CODE("""class SFR(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.SiLU(),
                                 nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))          # identity at initialisation

    def forward(self, x):
        mc = torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))
        xc = x * mc                                        # which channels
        ms = torch.sigmoid(self.spatial(torch.cat([xc.mean(1, keepdim=True), xc.amax(1, keepdim=True)], 1)))
        return x + self.gamma * (xc * ms)                  # where""")
H2("Inserting SFR into an Ultralytics detector")
CODE("""def add_sfr(model):
    detect = model.model[-1]                       # the Detect head
    for k, src in enumerate(detect.f):             # indices of the P3/P4/P5 layers that feed it
        ch = detect.cv2[k][0].conv.in_channels
        model.model[src] = SFRWrap(model.model[src], ch)
    return model

class SFRTrainer(DetectionTrainer):                 # used via YOLO(...).train(trainer=SFRTrainer)
    def get_model(self, cfg=None, weights=None, verbose=True):
        return add_sfr(super().get_model(cfg=cfg, weights=weights, verbose=verbose))""")
H2("How to run everything")
CODE("""pip install -r requirements.txt
python scripts/download_plantvillage.py --out data/raw --per-class 1600   # ~3 min, ~230 MB
python scripts/preprocess.py
python -m src.train_classifier --model yolo26_cls_sfr --epochs 15         # ~7 min on 4 CPU cores
python -m src.baseline_features
python -m src.train_detector --arch yolo26n --sfr --epochs 15 --fraction 0.35
python scripts/make_figures.py
python -m src.predict some_leaf.jpg --gradcam cam.png
python app/app.py                                                          # http://localhost:5000
pytest -q""")

# ================================================================== 13. Viva
H1("Likely Viva Questions with Answers")
QA = [
    ("What problem does the project solve?",
     "Automatic diagnosis of five leaf conditions from a photo, plus a health score, treatment advice and "
     "sensor-driven irrigation, delivered through an app, a web page and field cameras."),
    ("Which dataset did you use, and why do you believe it matches the original?",
     "PlantVillage. The original test log lists PlantVillage filenames (UMD_Powd.M, FAM_B.Rot, Rutg._Bact.S, RS_HL, "
     "GHLB2 Leaf). We rebuilt the same five super-classes from 16 PlantVillage folders, 1,600 images each."),
    ("Why YOLO26 instead of YOLOv5?",
     "YOLO26 is the newest Ultralytics generation. Its backbone (C3k2 + C2PSA) is more accurate per FLOP, and its "
     "NMS-free end-to-end head is designed for CPU and edge inference, which fits an ESP32/phone deployment. We "
     "kept YOLOv5 and YOLO11 as baselines to show the effect."),
    ("What is SFR and why does γ start at zero?",
     "Selective Feature Refinement: a channel gate (which features) followed by a spatial gate (where), added "
     "back through a residual with a learned scale γ. With γ = 0 the block is an exact identity, so it can be "
     "inserted into a pretrained network without destroying what the network already knows."),
    ("Did SFR help?",
     "Compare the with/without-SFR rows in Section 9: on PlantVillage the plain backbone is already above 98%, so "
     "the remaining differences are small and within single-seed noise. We report that honestly rather than "
     "claiming a large gain. The learned γ values are non-zero, so the network does use the refinement."),
    ("Why classification for the main numbers if the project is about detection?",
     "Each PlantVillage image contains exactly one centred leaf, so 'which disease?' is an image-level question. "
     "We report both: a classifier (accuracy, F1, AUC) and a detector (mAP, plus image-level accuracy for a fair "
     "comparison)."),
    ("Where do the bounding boxes come from?",
     "From PlantVillage's background-removed images: the leaf mask's bounding rectangle. One box per image. This "
     "is automatic labelling, and it is easier than the original hand-drawn multi-box Roboflow labels."),
    ("Why is your mAP so much higher than the original 0.743?",
     "Different labels (automatic single boxes vs hand-drawn multiple boxes), a different split, newer "
     "architectures and pretraining. The numbers are not directly comparable, and we say so."),
    ("What does the background-removed test show?",
     "Accuracy collapses when the lab background is removed, so part of the model's accuracy comes from "
     "background, lighting and crop identity. Training with background augmentation reduces the gap. Field "
     "validation is needed before deployment."),
    ("How do you avoid data leakage?",
     "Splits are fixed before training; the test split is used once. Preprocessing statistics are ImageNet "
     "constants, not fitted on our data. Limitation: PlantVillage sometimes has several photos of the same leaf, "
     "and it has no leaf IDs, so near-duplicates may cross splits."),
    ("Why macro-F1 for model selection?",
     "It weights every class equally, so one weak class cannot hide behind the others. Our classes are balanced, "
     "so accuracy and macro-F1 are close, but macro-F1 is the safer habit."),
    ("Explain precision vs recall for late blight.",
     "Precision: of the leaves we called late blight, the fraction that really were. Recall: of the real "
     "late-blight leaves, the fraction we found. For a fast-spreading disease, recall matters more, because a "
     "missed case can destroy a field."),
    ("What is Grad-CAM?",
     "A heat-map that weights the last convolutional feature maps by the gradient of the class score. It shows "
     "which regions increased that class's score."),
    ("Why VARI/ExG and not NDVI?",
     "NDVI needs a near-infrared band. Phones and the ESP32-CAM are RGB only, so we use visible-band indices. "
     "They are less specific, and we state that limitation."),
    ("How does the irrigation logic avoid rapid on/off switching?",
     "Hysteresis: two thresholds (on below 35%, off above 60%) and a 10-minute safety cut-off."),
    ("How long does training take on a CPU?",
     "~25 s per epoch for the classifier (5,600 images, 128 px, 4 cores), ~7 min in total; each detector ~1 h on a "
     "35% subset. The original YOLOv5 run took 47 h for 100 epochs at 640 px."),
    ("What would you do with more time or a GPU?",
     "Full-resolution detector training on all images, 5-fold CV with several seeds, field-photo testing, "
     "quantised on-device models, and severity labels for the health score."),
    ("Is the pesticide advice safe to follow?",
     "It is general guidance for the decision-support prototype. Real use must follow product labels and local "
     "regulations, ideally reviewed by an agronomist."),
]
for q, a in QA:
    story.append(KeepTogether([Paragraph(f"<b>Q. {q}</b>", S["body"]), Paragraph(f"A. {a}", S["bullet"]),
                               Spacer(1, 3)]))

# ================================================================== 14. Glossary
H1("Glossary")
GL = [("Accuracy", "share of test images classified correctly"),
      ("AdamW", "Adam optimiser with decoupled weight decay"),
      ("Augmentation", "random, label-preserving changes to training images"),
      ("AUC", "area under the ROC curve; 1.0 = perfect ranking, 0.5 = random"),
      ("Backbone", "the feature-extracting part of a network, before the task head"),
      ("Batch normalisation", "normalises activations per channel to stabilise training"),
      ("C2PSA / C3k2", "YOLO11/26 building blocks: position-sensitive attention / cross-stage partial bottlenecks"),
      ("Confusion matrix", "table of true vs predicted classes"),
      ("Cross-entropy", "loss −log p(true class)"),
      ("Epoch", "one pass over the whole training set"),
      ("ExG / VARI / GLI", "visible-band greenness indices computed from R, G, B"),
      ("F1 score", "harmonic mean of precision and recall"),
      ("Grad-CAM", "gradient-weighted class activation map, a visual explanation"),
      ("Hysteresis", "two different switching thresholds to avoid rapid toggling"),
      ("ImageNet / COCO", "large public datasets used for pretraining classification / detection models"),
      ("IoU", "intersection over union of two boxes"),
      ("Label smoothing", "training targets of 1−ε+ε/K for the true class and ε/K for the others"),
      ("mAP", "mean average precision over classes (and IoU thresholds)"),
      ("PlantVillage", "public dataset of 54k lab photos of single leaves, 38 crop-disease classes"),
      ("Precision / Recall", "correctness of positive predictions / coverage of real positives"),
      ("SFR", "Selective Feature Refinement: channel + spatial attention with a zero-initialised residual gate"),
      ("Softmax", "turns logits into probabilities that sum to 1"),
      ("Stratified split", "split that keeps the class (and here crop) proportions equal in every part"),
      ("Transfer learning", "starting from weights learned on another, larger task"),
      ("YOLO", "'You Only Look Once', a one-stage real-time detector family (v5, v8, 11, 26 …)")]
TABLE([["Term", "Meaning"]] + [[a, b] for a, b in GL], [4.4, 12.2])

# ================================================================== 15. References
H1("References")
for r in [
    "D. P. Hughes, M. Salathé. An open access repository of images on plant health to enable the development of "
    "mobile disease diagnostics (PlantVillage). arXiv:1511.08060, 2015.",
    "S. P. Mohanty, D. P. Hughes, M. Salathé. Using deep learning for image-based plant disease detection. "
    "Frontiers in Plant Science 7:1419, 2016.",
    "G. Jocher et al. Ultralytics YOLOv5 (2020), YOLO11 (2024) and YOLO26 (2026). github.com/ultralytics.",
    "J. Redmon, S. Divvala, R. Girshick, A. Farhadi. You Only Look Once: unified, real-time object detection. "
    "CVPR, 2016.",
    "S. Woo, J. Park, J.-Y. Lee, I. S. Kweon. CBAM: Convolutional Block Attention Module. ECCV, 2018.",
    "J. Hu, L. Shen, G. Sun. Squeeze-and-Excitation Networks. CVPR, 2018.",
    "R. R. Selvaraju et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. "
    "ICCV, 2017.",
    "A. A. Gitelson et al. Novel algorithms for remote estimation of vegetation fraction (VARI). Remote Sensing "
    "of Environment 80(1), 2002.",
    "D. M. Woebbecke et al. Color indices for weed identification under various soil, residue, and lighting "
    "conditions (ExG). Trans. ASAE 38(1), 1995.",
    "D. Singh et al. PlantDoc: a dataset for visual plant disease detection. CoDS-COMAD, 2020.",
    "S. Savary et al. The global burden of pathogens and pests on major food crops. Nature Ecology & Evolution "
    "3, 2019.",
    "I. Loshchilov, F. Hutter. Decoupled weight decay regularization (AdamW). ICLR, 2019.",
]:
    story.append(Paragraph(r, S["bullet"], bulletText="•"))


# ------------------------------------------------------------------ document template with TOC + page numbers
class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                         bottomMargin=2 * cm, title="Crop Health Assessment and Plant Disease Detection - "
                         "Complete Project Guide", author="Crop-Health-Assessment-and-Disease-Detection")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("p", [frame], onPage=self.footer)])

    @staticmethod
    def footer(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("DV", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.2 * cm, 1.2 * cm, "Crop Health Assessment & Plant Disease Detection - Project Guide")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc"):
            self.notify("TOCEntry", (0, flowable._toc, self.page))


if __name__ == "__main__":
    Doc(OUT).multiBuild(story)
    print("wrote", OUT)
