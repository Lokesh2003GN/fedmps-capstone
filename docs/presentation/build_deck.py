#!/usr/bin/env python3
"""FedMPS final-review presentation: 14 slides, four sections of three.

    title | 01 Problem and background | 02 Data and system design |
    03 Federated training, privacy and application | 04 Results and conclusion | thank you

Every number comes from the thesis (docs/FedMPS_Capstone_Thesis_v3) and the
results/ files; figures are the thesis figures. Speaker notes carry the talk.

Usage:
    python docs/presentation/build_deck.py
"""
import copy
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "thesis" / "figures"
OUT = Path(__file__).resolve().parent / "FedMPS_Presentation.pptx"

# Palette from the thesis figures: deep teal dominates, brown is the accent.
DARK = RGBColor(0x0B, 0x3D, 0x37)      # title and closing slides
TEAL = RGBColor(0x0F, 0x5F, 0x56)
TINT = RGBColor(0xE8, 0xF2, 0xEF)      # card fill on white slides
BROWN = RGBColor(0x8A, 0x5A, 0x12)
SAND = RGBColor(0xF6, 0xEE, 0xE1)      # accent card fill
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x55, 0x5F, 0x5D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
TITLE_ONLY, BLANK = prs.slide_layouts[5], prs.slide_layouts[6]
SECTIONS = []          # (section title, [slides])
slide_no = 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def run(p, text, size, color=INK, bold=False, italic=False, font=BODY):
    r = p.add_run()
    r.text = text
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.color.rgb = color
    return r


def textbox(slide, x, y, w, h, name, anchor=MSO_ANCHOR.TOP, margin=0.05):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    return tf


def para(tf, first=False, align=PP_ALIGN.LEFT, after=0, before=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after, p.space_before = Pt(after), Pt(before)
    return p


def bullets(tf, items, size=16, color=INK, after=8, bold_lead=True):
    """Items are 'Lead: rest' strings; the lead is bold. A real bullet, not a glyph."""
    for k, item in enumerate(items):
        p = para(tf, first=(k == 0), after=after)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Inches(0.28)))
        pPr.set("indent", str(-Inches(0.28)))
        bu = etree.SubElement(pPr, qn("a:buChar"))
        bu.set("char", "•")
        lead, _, rest = item.partition(": ")
        if rest and bold_lead:
            run(p, lead + ": ", size, color, bold=True)
            run(p, rest, size, color)
        else:
            run(p, item, size, color)


def shape(slide, kind, x, y, w, h, fill, name, line=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    s.shadow.inherit = False
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def badge(slide, x, y, label, fill=TEAL, d=0.62, size=18):
    """The deck's motif: a number or short label in a filled circle."""
    c = shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill, f"Badge {label}")
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = para(tf, first=True, align=PP_ALIGN.CENTER)
    run(p, label, size, WHITE, bold=True, font=HEAD)
    return c


def card(slide, x, y, w, h, heading, body, label=None, fill=TINT, head_color=TEAL,
         size=15, name="Card"):
    shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, f"{name} background")
    left = x + 0.25
    if label:
        badge(slide, x + 0.25, y + 0.25, label, fill=head_color, d=0.55, size=16)
        left = x + 0.95
    tf = textbox(slide, left, y + 0.22, x + w - left - 0.2, 0.6, f"{name} heading",
                 anchor=MSO_ANCHOR.MIDDLE)
    run(para(tf, first=True), heading, 18, head_color, bold=True, font=HEAD)
    tf = textbox(slide, x + 0.25, y + 0.95, w - 0.45, h - 1.1, f"{name} text")
    if isinstance(body, list):
        bullets(tf, body, size=size, after=6)
    else:
        run(para(tf, first=True), body, size, INK)


def stat(slide, x, y, w, value, label, color=TEAL, vsize=40, name="Stat"):
    tf = textbox(slide, x, y, w, 0.85, f"{name} value", margin=0)
    run(para(tf, first=True), value, vsize, color, bold=True, font=HEAD)
    tf = textbox(slide, x, y + 0.85, w, 0.75, f"{name} label", margin=0)
    run(para(tf, first=True), label, 13, MUTED)


def picture(slide, path, x, y, w=None, h=None, name="Figure"):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   Inches(w) if w else None, Inches(h) if h else None)
    pic.name = name
    return pic


def caption(slide, x, y, w, text, name="Caption"):
    tf = textbox(slide, x, y, w, 0.35, name, margin=0)
    run(para(tf, first=True), text, 11, MUTED, italic=True)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def content_slide(section_no, section_title, title):
    """White slide, section label and title at the same place on every content slide."""
    global slide_no
    slide = prs.slides.add_slide(TITLE_ONLY)
    slide_no += 1
    SECTIONS[-1][1].append(slide)
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.62), Inches(12.1), Inches(0.85)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, title, 32, INK, bold=True, font=HEAD)
    lab = textbox(slide, 0.6, 0.28, 8, 0.32, "Section label", margin=0)
    run(para(lab, first=True), f"{section_no}  ·  {section_title.upper()}", 12, BROWN, bold=True)
    foot = textbox(slide, 0.6, 7.02, 8, 0.3, "Footer", margin=0)
    run(para(foot, first=True), "FedMPS  ·  Privacy-preserving federated missing-person search",
        10, MUTED)
    num = textbox(slide, 11.73, 7.02, 1.0, 0.3, "Slide number", margin=0)
    run(para(num, first=True, align=PP_ALIGN.RIGHT), str(slide_no), 10, MUTED)
    return slide


def dark_slide(section):
    global slide_no
    slide = prs.slides.add_slide(BLANK)
    slide_no += 1
    SECTIONS.append((section, [slide])) if not SECTIONS or SECTIONS[-1][0] != section \
        else SECTIONS[-1][1].append(slide)
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = DARK
    return slide


def section(title):
    SECTIONS.append((title, []))


# --------------------------------------------------------------------------
# 1  Title
# --------------------------------------------------------------------------
s = dark_slide("Title")
badge(s, 0.9, 0.95, "FL", fill=BROWN, d=0.9, size=24)
tf = textbox(s, 0.9, 2.05, 11.8, 1.7, "Title")
run(para(tf, first=True), "FedMPS: Privacy-Preserving Federated Learning", 36, WHITE,
    bold=True, font=HEAD)
run(para(tf), "for Missing Person Identification", 36, WHITE, bold=True, font=HEAD)
tf = textbox(s, 0.9, 3.85, 11.5, 0.5, "Subtitle")
run(para(tf, first=True), "Searching CCTV across camera sites without pooling footage or faces",
    20, RGBColor(0xBF, 0xDD, 0xD6), italic=True)
tf = textbox(s, 0.9, 4.9, 7.4, 1.7, "Team")
for k, (name, reg) in enumerate([("G N Lokesh", "23BCE9603"), ("R Kumaresh", "23BCE9585"),
                                 ("K Kishore", "23BCE9746"), ("Aswin Maheswaran", "23BCE8540")]):
    p = para(tf, first=(k == 0), after=2)
    run(p, f"{name}  ", 16, WHITE, bold=True)
    run(p, reg, 16, RGBColor(0xBF, 0xDD, 0xD6))
tf = textbox(s, 8.3, 4.9, 4.4, 1.7, "Guide and school")
run(para(tf, first=True, after=2), "Guide: Kanhaiya Sharma", 16, WHITE, bold=True)
run(para(tf, after=2), "School of Computer Science and Engineering", 14, RGBColor(0xBF, 0xDD, 0xD6))
run(para(tf, after=2), "VIT-AP University  ·  Capstone final review", 14, RGBColor(0xBF, 0xDD, 0xD6))
notes(s, "Good morning. We are presenting FedMPS, a privacy-preserving federated learning system "
         "for finding a missing person in CCTV footage held by several camera sites. The talk has "
         "four parts: the problem and related work, the data and system design, federated "
         "training with its privacy mechanisms and the application, and finally the results.")

# --------------------------------------------------------------------------
# Section 01  Problem and background
# --------------------------------------------------------------------------
section("01 Problem and background")
S1 = ("01", "Problem and background")

s = content_slide(*S1, "Central CCTV search pools every bystander's face")
card(s, 0.6, 1.75, 3.9, 3.4, "Privacy risk",
     "A central server holding all footage copies the face of every bystander, becomes a single "
     "point of failure, and conflicts with data-minimisation rules. Sites are often not allowed "
     "to share footage at all.", label="1", name="Privacy card")
card(s, 4.72, 1.75, 3.9, 3.4, "Poor CCTV faces",
     "Faces on CCTV are small, blurred, compressed and badly lit. Accuracy drops sharply compared "
     "with the clean photograph that a family gives the police.", label="2", name="Quality card")
card(s, 8.84, 1.75, 3.9, 3.4, "Open-set search",
     "Almost everyone a camera sees is on no watchlist, so the system must stay silent for them. "
     "Closed-set accuracy, which most papers report, does not show this.", label="3",
     name="Openset card")
tf = textbox(s, 0.6, 5.55, 12.1, 0.9, "Question")
run(para(tf, first=True), "Our question: ", 17, BROWN, bold=True)
run(tf.paragraphs[0], "can camera sites search together, and even improve a shared model, "
                      "without sharing raw video, face crops or unprotected biometric data?",
    17, INK, italic=True)
notes(s, "Cities already have thousands of CCTV cameras, and modern face recognition could search "
         "them quickly. The obvious design, collecting all footage centrally, exposes every "
         "bystander and creates one point of failure. On top of that, CCTV faces are poor quality "
         "and the task is open-set: the system must reject almost everyone it sees. That leads to "
         "our research question at the bottom of the slide.")

s = content_slide(*S1, "Two objectives, built and measured end to end")
card(s, 0.6, 1.75, 5.95, 2.45, "Objective 1: federated multi-camera search",
     "Detection, tracking, quality gating, five-point alignment and ArcFace embeddings at every "
     "camera node, with the recognition model improved by federated learning across nodes that "
     "never share their data.", label="1", name="Objective 1")
card(s, 6.78, 1.75, 5.95, 2.45, "Objective 2: privacy that is measured",
     "Secure aggregation, differential privacy on model updates and cancelable watchlist "
     "templates, each measured against the same training run without it.", label="2",
     name="Objective 2")
for k, (head, body) in enumerate([
        ("Federated learning", "Nodes exchange only model updates, never data"),
        ("Privacy mechanisms", "Masked updates, noise, keyed 64-byte templates"),
        ("Human in the loop", "An operator confirms every alert before anything leaves a site")]):
    x = 0.6 + k * 4.12
    badge(s, x, 4.65, ["FL", "SA", "HR"][k], fill=BROWN, d=0.7, size=16)
    tf = textbox(s, x + 0.85, 4.6, 3.1, 0.45, f"Idea {k} heading", margin=0)
    run(para(tf, first=True), head, 17, INK, bold=True, font=HEAD)
    tf = textbox(s, x + 0.85, 5.05, 3.1, 1.0, f"Idea {k} text", margin=0)
    run(para(tf, first=True), body, 14, MUTED)
caption(s, 0.6, 6.35, 12.1, "Evaluated on 628,699 VGGFace2-HQ images degraded to CCTV quality and on "
                            "real CCTV footage from ChokePoint, and demonstrated in a working "
                            "application")
notes(s, "We set two objectives. First, a complete multi-camera search pipeline whose model is "
         "trained by federated learning. Second, privacy mechanisms whose cost we actually "
         "measure, not just claim. The system combines three ideas: federated learning, privacy "
         "mechanisms, and a human operator who must confirm every match.")

s = content_slide(*S1, "Sixteen papers surveyed, five gaps addressed")
tf = textbox(s, 0.6, 1.75, 5.6, 0.45, "Groups heading", margin=0)
run(para(tf, first=True), "What the literature covered", 20, TEAL, bold=True, font=HEAD)
for k, (n, head, body) in enumerate([
        ("3", "Missing-person search and templates", "YOLO with ArcFace, 97.5%; keyed 96-bit "
                                                     "templates; privacy survey"),
        ("4", "Federated face recognition", "Split FL with SMPC, FedProx on non-IID LFW, "
                                            "blockchain FL, secure aggregator"),
        ("9", "Other facial tasks", "Expressions, attendance, edge video, de-identification, "
                                    "augmentation")]):
    y = 2.35 + k * 1.3
    badge(s, 0.6, y, n, d=0.75, size=22)
    tf = textbox(s, 1.55, y - 0.05, 4.7, 0.45, f"Group {k} heading", margin=0)
    run(para(tf, first=True), head, 16, INK, bold=True)
    tf = textbox(s, 1.55, y + 0.38, 4.7, 0.8, f"Group {k} text", margin=0)
    run(para(tf, first=True), body, 13, MUTED)
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 6.7, 1.75, 6.03, 4.95, SAND, "Gaps background")
tf = textbox(s, 6.95, 1.95, 5.6, 0.45, "Gaps heading", margin=0)
run(para(tf, first=True), "Gaps that FedMPS addressed", 20, BROWN, bold=True, font=HEAD)
tf = textbox(s, 6.95, 2.55, 5.6, 4.0, "Gaps")
bullets(tf, ["Single frames: decisions per frame, not per tracked person",
             "Closed-set accuracy: no open-set search with people off the watchlist",
             "Unmeasured privacy: no cost of each mechanism against the same run",
             "Missing bounds: no local-only and centralised baselines on identical data",
             "No oversight: matches disclosed without human review or audit"], size=15, after=9)
notes(s, "We surveyed sixteen recent papers in three groups. Across them we found five gaps: "
         "frame-by-frame decisions, closed-set evaluation, privacy claims without measured cost, "
         "federated results without proper baselines, and no human oversight. Each gap maps to a "
         "design decision you will see in the next sections.")

# --------------------------------------------------------------------------
# Section 02  Data and system design
# --------------------------------------------------------------------------
section("02 Data and system design")
S2 = ("02", "Data and system design")

s = content_slide(*S2, "VGGFace2-HQ, split 70/30 by identity across five nodes")
stat(s, 0.6, 1.8, 3.0, "628,699", "images, every image of the dataset", name="Images")
stat(s, 3.6, 1.8, 2.6, "4,605", "identities, 9 to 414 images each", name="Identities")
stat(s, 0.6, 3.55, 3.0, "3,223", "people trained across 5 nodes, 70%", color=BROWN, name="Train")
stat(s, 3.6, 3.55, 2.6, "1,382", "people held out, 30%, never trained on", color=BROWN,
     name="Held out")
tf = textbox(s, 0.6, 5.3, 5.6, 1.5, "Dataset notes")
bullets(tf, ["Split by identity: no person at two nodes or in both training and test",
             "Held out: a 691-person watchlist and 691 people as the crowd"], size=14, after=6)
cd = CategoryChartData()
cd.categories = ["Node 0", "Node 1", "Node 2", "Node 3", "Node 4", "Held out"]
cd.add_series("Images", (42678, 67577, 90004, 106409, 131385, 190646))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.6), Inches(1.7), Inches(6.13),
                        Inches(4.75), cd)
gf.name = "Images per node chart"
ch = gf.chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "Images per node and held-out set"
tp = ch.chart_title.text_frame.paragraphs[0]
tp.runs[0].font.size, tp.runs[0].font.name, tp.runs[0].font.bold = Pt(14), BODY, True
tp.runs[0].font.color.rgb = INK
plot = ch.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = "0\\,000", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.name, dl.font.color.rgb = Pt(11), BODY, INK
ser = plot.series[0]
for k in range(6):
    pt = ser.points[k]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = BROWN if k == 5 else TEAL
va = ch.value_axis
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE3, 0xE1)
va.format.line.fill.background()
va.tick_labels.font.size, va.tick_labels.font.color.rgb = Pt(11), MUTED
va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
ca = ch.category_axis
ca.tick_labels.font.size, ca.tick_labels.font.color.rgb = Pt(12), INK
ca.format.line.color.rgb = RGBColor(0xBB, 0xC4, 0xC2)
caption(s, 6.6, 6.5, 6.1, "Source: kaggle.com/datasets/hearfool/vggface2; VGGFace2-HQ restored copy")
notes(s, "We used every image of VGGFace2-HQ, 628,699 images of 4,605 people. The split is by "
         "identity, seventy percent of the people over five unequal nodes and thirty percent held "
         "out, so no test person is ever seen in training. The held-out people form a 691-person "
         "watchlist and an equal crowd of people who must not trigger alerts.")

s = content_slide(*S2, "Preprocessing: align, degrade to CCTV quality, embed")
for k, (head, body) in enumerate([
        ("Align", "SCRFD finds five landmarks and warps each face onto the 112 × 112 ArcFace "
                  "template; 0.08% could not be aligned"),
        ("Degrade", "Blur, 20 to 48 pixel faces, gamma, noise and JPEG 25 to 70, with one fixed "
                    "camera profile per node"),
        ("Embed", "Frozen ArcFace gives a 512-d embedding of the clean and the degraded copy; "
                  "84 minutes on an AWS T4 GPU")]):
    y = 1.8 + k * 1.62
    badge(s, 0.6, y, str(k + 1), d=0.62)
    tf = textbox(s, 1.42, y - 0.02, 4.6, 0.45, f"Step {k} heading", margin=0)
    run(para(tf, first=True), head, 19, INK, bold=True, font=HEAD)
    tf = textbox(s, 1.42, y + 0.42, 4.6, 1.1, f"Step {k} text", margin=0)
    run(para(tf, first=True), body, 14, MUTED)
picture(s, FIG / "aligned_crops_vgghq.png", 6.45, 1.55, h=5.15, name="Aligned crops figure")
caption(s, 6.45, 6.72, 6.2, "Top: enrolment crops and CCTV probes. Bottom: three faces through the "
                            "five camera profiles")
notes(s, "Every face is first aligned with SCRFD onto ArcFace's template, then degraded with the "
         "camera profile of its node, then embedded by the frozen backbone. Each node has a fixed "
         "profile: node 0 sees 21-pixel faces with heavy blur, node 4 sees 47-pixel faces, so the "
         "nodes are genuinely non-IID, which is the setting where federated learning should help.")

s = content_slide(*S2, "System architecture")
picture(s, FIG / "system_architecture_v2.png", 0.6, 1.6, h=5.3, name="System architecture figure")
tf = textbox(s, 8.7, 1.65, 4.05, 5.2, "Architecture points")
bullets(tf, ["Camera nodes: detect, track, quality-gate, embed and match on site",
             "Cloud: coordinator aggregates masked adapter updates; GPU instance prepares data "
             "and evaluates",
             "Privacy modules: DP, cancelable templates, secure aggregation",
             "Dashboard: server console, node console, confirmed sightings, audit log",
             "Network carries only: keyed templates, masked updates, one confirmed crop"],
        size=14, after=8)
notes(s, "This is the full architecture. On the left, each camera node runs the same local "
         "pipeline on its own video. In the middle, the cloud side coordinates federated training "
         "and holds models and logs. Only three kinds of data ever cross the network: keyed "
         "templates down to the nodes, masked adapter updates up to the server, and a single face "
         "crop after an operator confirms an alert.")

# --------------------------------------------------------------------------
# Section 03  Federated training, privacy and application
# --------------------------------------------------------------------------
section("03 Federated training, privacy and application")
S3 = ("03", "Federated training, privacy and application")

s = content_slide(*S3, "Federated training shares only a 1.05 MB adapter")
picture(s, FIG / "fl_architecture.png", 0.6, 1.6, h=5.25, name="FL architecture figure")
stat(s, 6.15, 1.75, 3.2, "263,424", "adapter parameters on a frozen 43.6 M ArcFace", name="Params")
stat(s, 9.55, 1.75, 3.2, "165×", "smaller update than federating the backbone", color=BROWN,
     name="Smaller")
tf = textbox(s, 6.15, 3.65, 6.6, 3.1, "Training points")
bullets(tf, ["Loss: symmetric InfoNCE pulls a degraded face to its clean embedding",
             "FedProx: proximal term, μ = 0.3, limits drift between non-IID nodes",
             "FedAvg: server averages updates weighted by node size",
             "Schedule: 200 rounds, 15% of each node's data per round, cosine learning rate",
             "Selection: settings fixed in advance by a pilot on training identities only"],
        size=14, after=6)
notes(s, "We keep ArcFace frozen and federate only a small residual adapter of 263 thousand "
         "parameters, so each update is 1.05 megabytes, 165 times less than sending the backbone. "
         "Each node trains with a contrastive InfoNCE loss plus the FedProx term, and the server "
         "averages the updates. All settings were fixed by a pilot before the main 200-round run, "
         "so nothing was tuned on the test set.")

s = content_slide(*S3, "Four privacy safeguards between a camera and the server")
items = [("Secure aggregation", "Pairwise Diffie–Hellman masks cancel in the sum, so the server "
                                "sees only the total of the five updates", "SA"),
         ("Differential privacy", "Each update clipped to C = 0.313 and Gaussian noise added; "
                                  "privacy budget tracked with a Rényi accountant", "DP"),
         ("Cancelable templates", "512-bit keyed code per case and node, 64 bytes: revocable, "
                                  "unlinkable across sites, harder to invert", "CT"),
         ("Human review and audit", "Alerts stay pending at the node; only a confirmed crop "
                                    "leaves, and every action is logged", "HR")]
for k, (head, body, lab) in enumerate(items):
    x = 0.6 + (k % 2) * 6.18
    y = 1.75 + (k // 2) * 2.55
    card(s, x, y, 5.95, 2.3, head, body, label=lab,
         fill=SAND if k == 3 else TINT, head_color=BROWN if k == 3 else TEAL, name=f"Safeguard {k}")
notes(s, "Four safeguards protect the people involved. Secure aggregation hides each site's "
         "update from the server. Differential privacy adds calibrated noise to updates. The "
         "watchlist is sent only as keyed templates that differ per case and per site, and can be "
         "revoked. And a human operator must confirm every alert before even one face crop leaves "
         "a site, with everything logged.")

s = content_slide(*S3, "At the node: from camera frame to a reviewed alert")
picture(s, FIG / "node_pipeline.png", 0.6, 1.65, w=6.1, name="Node pipeline figure")
tf = textbox(s, 0.6, 5.0, 6.1, 1.85, "Pipeline points")
bullets(tf, ["Quality gate: drops tiny, blurred or side-on faces before they cause false alarms",
             "Track vote: an alert only when 3 of the last 5 frames of one person agree"],
        size=14, after=6)
picture(s, FIG / "v2_node_review.png", 7.05, 1.65, h=4.35, name="Node review screenshot")
caption(s, 7.05, 6.08, 5.7, "Node 0 console: the operator compares the reported photograph with "
                            "the camera crop and confirms or rejects")
notes(s, "At each node, sampled frames go through detection, tracking and a quality gate, then "
         "the backbone and adapter, matching and a three-of-five track vote. The result is a "
         "pending alert that only that site's operator can see. On the right is our node console "
         "from the demo on the university event video: the operator compares the reported "
         "photograph with what the camera saw, and confirms or rejects.")

# --------------------------------------------------------------------------
# Section 04  Results and conclusion
# --------------------------------------------------------------------------
section("04 Results and conclusion")
S4 = ("04", "Results and conclusion")

s = content_slide(*S4, "Federated matched pooling and beat training alone")
cd = CategoryChartData()
cd.categories = ["Local only", "Frozen backbone", "Centralised", "Federated"]
cd.add_series("TPIR at 1% FPIR", (0.8812, 0.9054, 0.9080, 0.9107))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.65), Inches(7.0),
                        Inches(4.6), cd)
gf.name = "TPIR chart"
ch = gf.chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "TPIR at 1% FPIR on 691-person watchlist, axis from 0.85"
tp = ch.chart_title.text_frame.paragraphs[0]
tp.runs[0].font.size, tp.runs[0].font.name, tp.runs[0].font.bold = Pt(14), BODY, True
tp.runs[0].font.color.rgb = INK
plot = ch.plots[0]
plot.gap_width = 55
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = "0.0000", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.name, dl.font.color.rgb = Pt(12), BODY, INK
for k in range(4):
    pt = plot.series[0].points[k]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = TEAL if k == 3 else RGBColor(0x9D, 0xB8, 0xB2)
va = ch.value_axis
va.minimum_scale, va.maximum_scale = 0.85, 0.92
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE3, 0xE1)
va.format.line.fill.background()
va.tick_labels.font.size, va.tick_labels.font.color.rgb = Pt(11), MUTED
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0.00", False
ca = ch.category_axis
ca.tick_labels.font.size, ca.tick_labels.font.color.rgb = Pt(13), INK
ca.format.line.color.rgb = RGBColor(0xBB, 0xC4, 0xC2)
stat(s, 8.0, 1.75, 2.3, "0.9734", "rank-1, federated; centralised 0.9733", name="Rank1")
stat(s, 10.45, 1.75, 2.3, "+1.1", "rank-1 points over training alone", color=BROWN, name="Gain")
tf = textbox(s, 8.0, 3.65, 4.75, 3.0, "Result points")
bullets(tf, ["Pooling: keeping data at the sites cost no accuracy",
             "Alone: every site's own adapter fell below the frozen backbone",
             "Adapter: small, stable gain over 200 rounds, below our 2-point bar"],
        size=14, after=8)
caption(s, 0.6, 6.35, 7.0, "Same test set and scoring for every arm; 30,540 watchlist probes, "
                           "31,850 crowd probes")
notes(s, "This is the main result. On the open-set metric, TPIR at one percent false positive "
         "identification rate, the federated adapter reached 0.9107, equal to centralised training "
         "on pooled data within sampling noise, and above the frozen backbone. Training at one site "
         "alone was worse than not training. Note the axis starts at 0.85, so the differences are "
         "small: the gain of the adapter is real but below the two-point bar we set in advance.")

s = content_slide(*S4, "Privacy cost measured, and it works on real CCTV")
for k, (val, lab, col) in enumerate([
        ("0.0000", "rank-1 cost of secure aggregation", TEAL),
        ("0.501", "membership-inference AUC, chance level", TEAL),
        ("−4.7", "rank-1 points for 512-bit templates", BROWN),
        ("ε = 252", "even the weakest DP noise destroyed accuracy", BROWN)]):
    x = 0.6 + (k % 2) * 3.05
    y = 1.75 + (k // 2) * 2.05
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, 2.85, 1.85, TINT if col == TEAL else SAND,
          f"Privacy stat {k} background")
    stat(s, x + 0.2, y + 0.15, 2.5, val, lab, color=col, vsize=32, name=f"Privacy stat {k}")
tf = textbox(s, 0.6, 5.95, 5.9, 0.9, "Privacy note")
run(para(tf, first=True), "Secure aggregation and templates are usable; differential privacy has "
                          "no usable setting with only five sites.", 14, MUTED, italic=True)
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 6.95, 1.75, 5.78, 4.95, DARK, "CCTV panel")
tf = textbox(s, 7.25, 1.95, 5.2, 0.5, "CCTV heading", margin=0)
run(para(tf, first=True), "Real CCTV: ChokePoint, 29 people", 20, WHITE, bold=True, font=HEAD)
for k, (val, lab) in enumerate([("95.2%", "of enrolled walk-throughs alerted, 1,988 of 2,088"),
                                ("0", "alerts that named the wrong person"),
                                ("99.9%", "found and named at 1% FPIR, official crops")]):
    y = 2.7 + k * 1.3
    tf = textbox(s, 7.25, y, 2.2, 0.8, f"CCTV value {k}", margin=0)
    run(para(tf, first=True), val, 32, RGBColor(0xF2, 0xC9, 0x8B), bold=True, font=HEAD)
    tf = textbox(s, 9.5, y + 0.12, 3.0, 0.9, f"CCTV label {k}", margin=0)
    run(para(tf, first=True), lab, 14, WHITE)
notes(s, "Each privacy mechanism was measured against the same run without it. Secure aggregation "
         "cost exactly nothing. A corrected membership-inference attack scored 0.501, which is "
         "chance. 512-bit templates cost 4.7 rank-1 points. Differential privacy, honestly, had no "
         "usable setting with only five sites. On real CCTV footage from ChokePoint, the live "
         "pipeline with the latest adapter alerted on 95.2 percent of enrolled walk-throughs and "
         "never named the wrong person.")

s = content_slide(*S4, "Compared with published work, and what we conclude")
rows = [("Reference", "Metric", "Paper", "FedMPS"),
        ("Cao et al. 2018 [17]", "TAR @ FAR 10⁻⁴", "82.5%", "57.5%"),
        ("FedHide [18]", "EER", "10.4–12.9%", "11.8%"),
        ("FedFS [19]", "TPIR @ FPIR 10⁻³", "0.872", "0.836"),
        ("Kim, Kim & Teoh [20]", "TAR @ FAR 10⁻⁴", "57.1%", "57.5%"),
        ("DP-FedFace [21]", "TAR @ FAR 10⁻³", "87.8%", "68.4%"),
        ("Ji et al. 2022 [22]", "TAR @ FAR 10⁻⁴", "91.2%", "57.5%")]
gt = s.shapes.add_table(len(rows), 4, Inches(0.6), Inches(1.7), Inches(6.6), Inches(3.9))
gt.name = "Comparison table"
tbl = gt.table
for j, w in enumerate((2.45, 1.75, 1.2, 1.2)):
    tbl.columns[j].width = Inches(w)
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        c = tbl.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = TEAL if i == 0 else (TINT if i % 2 else WHITE)
        c.margin_left = c.margin_right = Inches(0.08)
        tf = c.text_frame
        tf.paragraphs[0].text = ""
        run(tf.paragraphs[0], v, 14, WHITE if i == 0 else INK, bold=(i == 0 or j == 3))
caption(s, 0.6, 5.72, 6.6, "Papers test on clean IJB images; FedMPS probes are 20–48 px CCTV faces, "
                           "so figures are indicative")
shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 7.55, 1.7, 5.18, 4.95, SAND, "Conclusion background")
tf = textbox(s, 7.8, 1.9, 4.7, 0.5, "Conclusion heading", margin=0)
run(para(tf, first=True), "Conclusion", 20, BROWN, bold=True, font=HEAD)
tf = textbox(s, 7.8, 2.45, 4.75, 4.1, "Conclusion points")
bullets(tf, ["Freeze the backbone: it does almost all of the work",
             "Federate the adapter: no cost against pooling, helps the worst cameras",
             "Never train alone: every single-site adapter lost accuracy",
             "Future: harder real footage, per-site thresholds, multiple seeds, dropout-robust "
             "secure aggregation"], size=16, after=12)
notes(s, "Against five recent papers and the VGGFace2 reference, FedMPS matches the federated and "
         "protected results on EER and TAR at ten to the minus four, even though our probes are "
         "CCTV-degraded rather than clean benchmark images, so the comparison is indicative. Our "
         "conclusion: freeze the backbone, federate a small adapter, and never fine-tune at a "
         "single site alone. Future work is harder real footage, per-site thresholds and more "
         "seeds.")

# --------------------------------------------------------------------------
# 14  Thank you
# --------------------------------------------------------------------------
s = dark_slide("Thank you")
badge(s, 6.21, 1.35, "?", fill=BROWN, d=0.9, size=30)
tf = textbox(s, 0.9, 2.5, 11.5, 1.2, "Thank you title")
run(para(tf, first=True, align=PP_ALIGN.CENTER), "Thank you", 54, WHITE, bold=True, font=HEAD)
tf = textbox(s, 0.9, 3.75, 11.5, 0.6, "Questions")
run(para(tf, first=True, align=PP_ALIGN.CENTER), "We welcome your questions", 22,
    RGBColor(0xBF, 0xDD, 0xD6), italic=True)
tf = textbox(s, 0.9, 5.2, 11.5, 1.0, "Team line")
run(para(tf, first=True, align=PP_ALIGN.CENTER),
    "G N Lokesh  ·  R Kumaresh  ·  K Kishore  ·  Aswin Maheswaran", 16, WHITE, bold=True)
run(para(tf, align=PP_ALIGN.CENTER), "Guide: Kanhaiya Sharma  ·  VIT-AP University", 14,
    RGBColor(0xBF, 0xDD, 0xD6))
notes(s, "Thank you. We are happy to take questions, and we can show the live consoles or any "
         "part of the code or results in more detail.")

# --------------------------------------------------------------------------
# PowerPoint sections: Title, four sections of three, Thank you
# --------------------------------------------------------------------------
sec_list = [("Title", [prs.slides[0]])] + [(t, sl) for t, sl in SECTIONS if t.startswith("0")] \
    + [("Thank you", [prs.slides[13]])]
# content_slide appended slides to the current section; rebuild the lists by position.
order = list(prs.slides)
sec_list = [("Title", order[0:1]), ("01 Problem and background", order[1:4]),
            ("02 Data and system design", order[4:7]),
            ("03 Federated training, privacy and application", order[7:10]),
            ("04 Results and conclusion", order[10:13]), ("Thank you", order[13:14])]
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
pres = prs.part._element
ext_lst = pres.find(qn("p:extLst"))
if ext_lst is None:
    ext_lst = etree.SubElement(pres, qn("p:extLst"))
ext = etree.SubElement(ext_lst, qn("p:ext"))
ext.set("uri", "{521415D9-36F7-43E2-AB2F-B90AF26B5E84}")
sl = etree.SubElement(ext, f"{{{P14}}}sectionLst", nsmap={"p14": P14})
ids = {s.slide_id: s for s in prs.slides}
for k, (title, slides) in enumerate(sec_list):
    sec = etree.SubElement(sl, f"{{{P14}}}section")
    sec.set("name", title)
    sec.set("id", "{%08X-0000-4000-8000-%012X}" % (0x5EC00000 + k, k))
    lst = etree.SubElement(sec, f"{{{P14}}}sldIdLst")
    for sld in slides:
        e = etree.SubElement(lst, f"{{{P14}}}sldId")
        e.set("id", str(sld.slide_id))

prs.save(OUT)
print(f"wrote {OUT} ({len(prs.slides)} slides)")
