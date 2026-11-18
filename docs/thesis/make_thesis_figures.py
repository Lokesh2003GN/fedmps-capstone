#!/usr/bin/env python3
"""Diagrams for the thesis, drawn in the palette of Figure 1.

    workflow.png        training phase and operational phase
    fl_round.png        one federated round with its optional privacy steps
    template_flow.png   how a cancelable watchlist template is made and matched
    node_pipeline.png   the operational pipeline at a node

Usage:
    python docs/thesis/make_thesis_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
TEAL, TEAL_BG = "#0f5f56", "#eef5f3"
BROWN, BROWN_BG = "#8a5a12", "#fbf4ea"
INK, MUTED = "#1a1a1a", "#555555"


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


PAD_X, PAD_Y, GAP = 0.14, 0.11, 0.08   # inches of clear space inside every box


def _extent(ax, t):
    """A text's drawn width and height, in data units (inches here)."""
    bb = t.get_window_extent(ax.figure.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x1 - x0, y1 - y0


def box(ax, x, y, w, h, title, body="", edge=TEAL, face="white", size=9.0, bold=True):
    """A rounded box with a bold title and an optional grey body below it.

    The two are measured once drawn and stacked as one block, centred in the
    box with GAP between them. If the block would come closer than PAD_X /
    PAD_Y to the border, both shrink together until it fits."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.6, edgecolor=edge, facecolor=face))
    cx = x + w / 2
    t = ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.25)
    b = (ax.text(cx, y + h / 2, body, ha="center", va="center", fontsize=size - 2,
                 color=MUTED, linespacing=1.3) if body else None)
    entry = {"t": t, "b": b, "size": size, "geom": (x, y, w, h)}
    entry["scale"] = _fit(ax, entry, 1.0)
    ax.__dict__.setdefault("_boxes", []).append(entry)


def _fit(ax, e, scale):
    """Largest scale <= `scale` at which the box's text keeps its padding;
    the text is left stacked and centred at that scale."""
    x, y, w, h = e["geom"]
    t, b = e["t"], e["b"]
    for _ in range(40):
        t.set_fontsize(e["size"] * scale)
        if b:
            b.set_fontsize((e["size"] - 2) * scale)
        tw, th = _extent(ax, t)
        bw, bh = _extent(ax, b) if b else (0.0, 0.0)
        total = th + (GAP + bh if b else 0.0)
        if max(tw, bw) <= w - 2 * PAD_X and total <= h - 2 * PAD_Y:
            break
        scale *= 0.97
    top = y + h / 2 + total / 2
    t.set_y(top - th / 2)
    if b:
        b.set_y(top - th - GAP - bh / 2)
    return scale


def save(fig, ax, name):
    """Give every box in the figure the same text scale - the smallest any
    box needed - so neighbouring boxes never differ in font size, then save."""
    boxes = ax.__dict__.get("_boxes", [])
    if boxes:
        common = min(e["scale"] for e in boxes)
        for e in boxes:
            _fit(ax, e, common)
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax, x, y, w, h, title, edge, face):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=2.0, edgecolor=edge, facecolor=face))
    ax.text(x + 0.18, y + h - 0.28, title, ha="left", va="center", fontsize=11.5,
            fontweight="bold", color=edge)


def arrow(ax, x1, y1, x2, y2, color="#333333", text="", side=0.12, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color))
    if text:
        ax.text((x1 + x2) / 2 + side, (y1 + y2) / 2, text, fontsize=8, color=color,
                ha="left" if side >= 0 else "right", va="center", linespacing=1.25)


def elbow(ax, points, color="#333333", lw=1.5, head=12):
    """An arrow made of straight horizontal and vertical segments.

    `points` are the corners in order; the last segment carries the head.
    The head segment starts exactly at the last corner (shrinkA=0), so the
    corners close without a gap."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, linewidth=lw, solid_joinstyle="miter",
            solid_capstyle="projecting")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=head, linewidth=lw, color=color,
                                 shrinkA=0))


def chain(ax, xs, y, w, h, items, edge, face="white"):
    for i, (x, (t, b)) in enumerate(zip(xs, items)):
        box(ax, x, y, w, h, t, b, edge=edge, face=face)
        if i:
            arrow(ax, xs[i - 1] + w + 0.02, y + h / 2, x - 0.02, y + h / 2)


# The thesis prints at one column, about 6.3 in wide, so the diagrams use three
# or four boxes per row and large text instead of one long row.
W3, H3 = 2.3, 1.25             # box size, three per row
COLS3 = [0.6, 3.3, 6.0]


def zrow(ax, xs, y, items, edge, size=12.0):
    """A row of boxes joined left to right."""
    for i, (x, (t, b)) in enumerate(zip(xs, items)):
        box(ax, x, y, W3, H3, t, b, edge=edge, size=size)
        if i:
            arrow(ax, xs[i - 1] + W3 + 0.02, y + H3 / 2, x - 0.02, y + H3 / 2)


def back(ax, y_from, y_to, color="#333333"):
    """The return arrow from the end of one row to the start of the next:
    down from the last box, left along the gap, down into the first box."""
    y_mid = (y_from + y_to + H3) / 2
    x_from, x_to = COLS3[2] + W3 / 2, COLS3[0] + W3 / 2
    elbow(ax, [(x_from, y_from - 0.02), (x_from, y_mid), (x_to, y_mid),
               (x_to, y_to + H3 + 0.02)], color=color)


def workflow():
    fig, ax = canvas(9.0, 11.4)
    panel(ax, 0.35, 5.75, 8.3, 5.5, "Training phase: dataset preparation on AWS GPU, federated training",
          TEAL, TEAL_BG)
    panel(ax, 0.35, 0.1, 8.3, 5.45, "Operational phase: consoles and camera video", BROWN, BROWN_BG)
    r1, r2, r3 = 8.95, 7.35, 5.95
    zrow(ax, COLS3, r1, [("VGGFace2-HQ", "4,605 identities\n628,699 images"),
                         ("Align", "SCRFD landmarks\n112×112 ArcFace crop"),
                         ("Degrade", "one camera profile\nper node")], TEAL)
    back(ax, r1, r2)
    zrow(ax, COLS3, r2, [("Embed", "frozen ArcFace\nclean + degraded"),
                         ("Partition", "70% over 5 nodes\n30% held out"),
                         ("Federated training", "FedProx, 200 rounds\nlast round kept")], TEAL)
    box(ax, COLS3[2], r3, W3, 0.95, "Trained adapter", "residual MLP", edge=TEAL, size=12)
    arrow(ax, COLS3[2] + W3 / 2, r2 - 0.02, COLS3[2] + W3 / 2, r3 + 0.97)

    o1, o2, o3 = 3.75, 2.15, 0.35
    zrow(ax, COLS3, o1, [("Open case", "photo read once\nthumbnail kept"),
                         ("Keyed templates", "one 64-byte code\nper node"),
                         ("Scan video", "detect, track,\nquality gate")], BROWN)
    back(ax, o1, o2)
    zrow(ax, COLS3, o2, [("Embed + match", "ArcFace + adapter\ncosine or template"),
                         ("Track vote", "3 of last 5 frames\ngive a pending alert"),
                         ("Human review", "confirm or reject\nat the node")], BROWN)
    box(ax, COLS3[1], o3, W3, 0.95, "Audit log", "every action recorded", edge=BROWN, size=12)
    box(ax, COLS3[2], o3, W3, 0.95, "Confirmed sighting", "one crop to server", edge=BROWN, size=12)
    arrow(ax, COLS3[2] + W3 / 2, o2 - 0.02, COLS3[2] + W3 / 2, o3 + 0.97)
    y_mid = (o2 + o3 + 0.95) / 2
    elbow(ax, [(COLS3[2] + 0.4, o2 - 0.02), (COLS3[2] + 0.4, y_mid),
               (COLS3[1] + W3 / 2, y_mid), (COLS3[1] + W3 / 2, o3 + 0.97)])
    # The trained adapter reaches every node's matcher: drawn down the left margin.
    elbow(ax, [(COLS3[2] - 0.02, r3 + 0.47), (0.15, r3 + 0.47),
               (0.15, o2 + H3 / 2), (COLS3[0] - 0.02, o2 + H3 / 2)],
          color=TEAL, lw=1.8, head=14)
    ax.text(COLS3[0], r3 + 0.62, "trained adapter, used by every node",
            fontsize=10.5, color=TEAL, ha="left", va="bottom")
    ax.set_xlim(0, 9.0)
    save(fig, ax, "workflow.png")


def fl_round():
    fig, ax = canvas(12, 6.6)
    # Server across the full width, so every arrow starts and ends on a box.
    panel(ax, 0.1, 4.5, 11.8, 1.95, "Server", BROWN, BROWN_BG)
    box(ax, 0.9, 4.65, 4.6, 1.2, "Aggregate",
        "w_g <- w_g + sum_k (n_k / n) * delta_k\npairwise masks cancel in the sum", edge=BROWN)
    box(ax, 6.5, 4.65, 4.6, 1.2, "Next round",
        "cosine learning-rate decay\n200 rounds, last round kept", edge=BROWN)
    arrow(ax, 5.52, 5.25, 6.48, 5.25)
    ax.add_patch(FancyBboxPatch((0.1, 0.9), 11.8, 2.05,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=2.0, edgecolor=TEAL, facecolor=TEAL_BG))
    ax.text(6.0, 1.1, "Five camera nodes: images, embeddings and labels never leave a node",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEAL)
    for k in range(5):
        x = 0.35 + k * 2.33
        box(ax, x, 1.35, 2.0, 1.4, f"Node {k}",
            "local training\nInfoNCE + FedProx\n15% of its data", edge=TEAL)
        arrow(ax, x + 0.55, 4.48, x + 0.55, 2.77, color=BROWN)
        arrow(ax, x + 1.45, 2.77, x + 1.45, 4.48, color=TEAL)
    # Legend below the nodes.
    arrow(ax, 0.6, 0.55, 0.6, 0.1, color=BROWN)
    ax.text(0.75, 0.33, "global adapter w_g, 1.05 MB, to every node", fontsize=9.5,
            color=BROWN, ha="left", va="center")
    arrow(ax, 6.1, 0.1, 6.1, 0.55, color=TEAL)
    ax.text(6.25, 0.33, "delta_k only, optional DP clip and noise, pairwise mask",
            fontsize=9.5, color=TEAL, ha="left", va="center")
    save(fig, ax, "fl_round.png")


def template_flow():
    fig, ax = canvas(9.0, 6.6)
    r1, r2, r3 = 5.2, 3.05, 0.4
    zrow(ax, COLS3, r1, [("Reference photo", "read once, in memory"),
                         ("ArcFace embedding", "512 numbers"),
                         ("Binarise", "each dimension against\nits mean: 512 bits")], BROWN)
    box(ax, COLS3[2], r2, W3, H3, "XOR with key k", "key k = SHA-256 of\nmaster, case, node k",
        edge=BROWN, size=12)
    arrow(ax, COLS3[2] + W3 / 2, r1 - 0.02, COLS3[2] + W3 / 2, r2 + H3 + 0.02, color=BROWN)
    box(ax, COLS3[0], r2, W3 + 2.65, H3, "Five 64-byte templates, one per node",
        "a different bit pattern at every node", edge=BROWN, size=12)
    arrow(ax, COLS3[2] - 0.02, r2 + H3 / 2, COLS3[0] + W3 + 2.67, r2 + H3 / 2, color=BROWN)
    zrow(ax, COLS3, r3, [("Live face at node k", "ArcFace + adapter"),
                         ("Binarise + XOR key k", "same key as the template"),
                         ("Hamming similarity", "s = 1 − 2 d / 512\nalert if s ≥ 0.2656")], TEAL)
    # Node k's template drops to the matcher: down, right, down.
    x_from, x_to, y_mid = COLS3[0] + W3 + 2.0, COLS3[2] + W3 / 2, r2 - 0.4
    elbow(ax, [(x_from, r2 - 0.02), (x_from, y_mid), (x_to, y_mid), (x_to, r3 + H3 + 0.02)],
          color=TEAL)
    ax.text(x_to - 0.15, y_mid - 0.15, "node k's template: the same\nkey on both sides cancels",
            fontsize=10.5, color=TEAL, ha="right", va="top", linespacing=1.25)
    save(fig, ax, "template_flow.png")


def node_pipeline():
    fig, ax = canvas(9.0, 4.6)
    w, h = 1.9, 1.75
    xs = [0.2 + i * 2.25 for i in range(4)]
    rows = [[("Frames", "sampled at\nup to 8 fps"),
             ("Detect", "SCRFD det_10g\n5 landmarks"),
             ("Track", "ByteTrack, one\ntrack per person"),
             ("Quality gate", "eyes ≥ 24 px, blur,\nexposure, yaw ≤ 45°")],
            [("Embed", "aligned crop\nArcFace + adapter"),
             ("Match", "cosine ≥ 0.24 or\ntemplate ≥ 0.2656"),
             ("Vote", "3 of the last 5\nframes of a track"),
             ("Pending alert", "crop and score for\nthe node's operator")]]
    ys = [2.6, 0.2]
    for y, items in zip(ys, rows):
        for i, (x, (tt, bb)) in enumerate(zip(xs, items)):
            box(ax, x, y, w, h, tt, bb, edge=TEAL, size=12)
            if i:
                arrow(ax, xs[i - 1] + w + 0.02, y + h / 2, x - 0.02, y + h / 2)
    y_mid = (ys[0] + ys[1] + h) / 2
    elbow(ax, [(xs[3] + w / 2, ys[0] - 0.02), (xs[3] + w / 2, y_mid),
               (xs[0] + w / 2, y_mid), (xs[0] + w / 2, ys[1] + h + 0.02)])
    save(fig, ax, "node_pipeline.png")


def camera(ax, cx, cy, color):
    """A small CCTV camera glyph: body, lens and wall mount."""
    ax.add_patch(FancyBboxPatch((cx - 0.30, cy - 0.11), 0.48, 0.22,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                linewidth=1.4, edgecolor=color, facecolor="white"))
    ax.add_patch(plt.Polygon([[cx + 0.18, cy + 0.08], [cx + 0.32, cy + 0.13],
                              [cx + 0.32, cy - 0.13], [cx + 0.18, cy - 0.08]],
                             closed=True, linewidth=1.4, edgecolor=color, facecolor=color))
    ax.plot([cx - 0.18, cx - 0.18, cx - 0.36], [cy - 0.11, cy - 0.25, cy - 0.25],
            color=color, linewidth=1.4)


def fl_architecture():
    D = 0.7   # everything above the "inside a node" panel sits this much higher
    fig, ax = canvas(9.0, 8.9 + D)
    # --- server ------------------------------------------------------------
    panel(ax, 0.35, 6.47 + D, 8.3, 2.35, "Central server: coordinates training, never sees data", BROWN, BROWN_BG)
    sw, sh, sy = W3, 1.3, 6.98 + D
    sx = COLS3
    items = [("Secure sum", "masked updates added;\npairwise masks cancel"),
             ("FedAvg update", r"$w_g \leftarrow w_g + \sum_k \frac{n_k}{n}\,\Delta_k$"),
             ("Model selection", "settings from a pilot;\nlast round kept")]
    for i, (x, (t, b)) in enumerate(zip(sx, items)):
        box(ax, x, sy, sw, sh, t, b, edge=BROWN, size=12.5)
        if i:
            arrow(ax, sx[i - 1] + sw + 0.02, sy + sh / 2, x - 0.02, sy + sh / 2, color=BROWN)
    ax.text(4.5, 6.72 + D, r"Rényi DP accountant: privacy budget $\varepsilon$ over the rounds at $\delta = 10^{-5}$",
            ha="center", va="center", fontsize=10.5, color=BROWN)

    # --- network band ---------------------------------------------------------
    cw, cg, x0 = 1.4, 0.2, 0.6
    cx = [x0 + k * (cw + cg) for k in range(5)]
    top, bot = 6.42 + D, 5.62 + D
    for x in cx:
        arrow(ax, x + 0.42, top, x + 0.42, bot, color=BROWN, lw=1.8)
        arrow(ax, x + 0.98, bot, x + 0.98, top, color=TEAL, lw=1.8)

    # --- the five nodes -------------------------------------------------------
    ids = [322, 483, 644, 805, 969]
    ny, nh = 4.22 + D, 1.35
    for k, x in enumerate(cx):
        ax.add_patch(FancyBboxPatch((x, ny), cw, nh, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    linewidth=1.8, edgecolor=TEAL, facecolor=TEAL_BG))
        camera(ax, x + cw / 2, ny + nh - 0.3, TEAL)
        ax.text(x + cw / 2, ny + 0.62, f"Node {k}", ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=INK)
        ax.text(x + cw / 2, ny + 0.24, f"{ids[k]} people", ha="center", va="center",
                fontsize=10, color=MUTED)
    # Pairwise key agreement between nodes, drawn as dashed links under the cards.
    ky = ny - 0.28
    ax.plot([cx[0] + cw / 2, cx[4] + cw / 2], [ky, ky], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.3)
    for x in cx:
        ax.plot([x + cw / 2, x + cw / 2], [ny - 0.02, ky], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.3)
    ax.text(4.5, ky - 0.3, "pairwise Diffie–Hellman keys between every pair of nodes, used for the masks",
            ha="center", va="center", fontsize=10, color=MUTED)
    # Legend for the arrows between the server and the nodes.
    ly = ky - 0.68
    ax.text(1.75, ly, r"$\downarrow$ global adapter $w_g$", ha="center", va="center", fontsize=10.5,
            color=BROWN, fontweight="bold")
    ax.text(4.25, ly, r"$\uparrow$ masked update $\Delta_k$", ha="center", va="center", fontsize=10.5,
            color=TEAL, fontweight="bold")
    ax.text(6.95, ly, "1.05 MB each way, 200 rounds", ha="center", va="center", fontsize=10.5, color=INK)

    # --- inside a node --------------------------------------------------------
    panel(ax, 0.35, 0.1, 8.3, 3.55, "Inside each node, every round", TEAL, "white")
    bh, r1, r2 = 1.15, 1.95, 0.35
    rows = [(r1, [("Local data", "video, crops, labels:\nnever leave the node"),
                  ("Frozen ArcFace", "512-d embeddings,\nnot trained"),
                  ("Train adapter", "InfoNCE + FedProx,\nμ = 0.3, 15% of data")]),
            (r2, [("Update", r"$\Delta_k = w_k - w_g$"),
                  ("DP", "clip to C = 0.313,\nadd Gaussian noise"),
                  ("SA", "add pairwise masks,\nsend to server")])]
    for y, row in rows:
        for i, (x, (t, b)) in enumerate(zip(COLS3, row)):
            box(ax, x, y, W3, bh, t, b, edge=TEAL, size=12)
            if i:
                arrow(ax, COLS3[i - 1] + W3 + 0.02, y + bh / 2, x - 0.02, y + bh / 2)
    y_mid = (r1 + r2 + bh) / 2
    elbow(ax, [(COLS3[2] + W3 / 2, r1 - 0.02), (COLS3[2] + W3 / 2, y_mid),
               (COLS3[0] + W3 / 2, y_mid), (COLS3[0] + W3 / 2, r2 + bh + 0.02)], color=TEAL)
    save(fig, ax, "fl_architecture.png")


def system_architecture():
    """The whole system: the server, the network, and what runs at every node."""
    fig, ax = canvas(9.0, 10.6)
    W4, H4 = 1.72, 1.45
    X4 = [0.6 + i * 2.05 for i in range(4)]   # 0.33 in between boxes for the arrows

    # --- server ------------------------------------------------------------
    panel(ax, 0.35, 7.45, 8.3, 3.05, "Central server: investigator side", BROWN, BROWN_BG)
    items = [("Server\nconsole", "open, list and\nclose cases"),
             ("Template\nissuer", "photo read once;\n64-byte keyed\ncode per node"),
             ("FL\naggregator", "FedAvg of the\nmasked updates"),
             ("Store and\naudit log", "cases, templates,\nconfirmed alerts")]
    for x, (t, b) in zip(X4, items):
        box(ax, x, 7.75, W4, 2.1, t, b, edge=BROWN, size=12)

    # --- network -------------------------------------------------------------
    for x, up, col, label in ((1.1, False, BROWN, "keyed template\nper case"),
                              (3.1, False, BROWN, "global adapter\nevery round"),
                              (5.0, True, TEAL, "masked adapter\nupdate, 1.05 MB"),
                              (6.9, True, TEAL, "one face crop,\nonly if confirmed")):
        y1, y2 = (6.7, 7.4) if up else (7.4, 6.7)
        arrow(ax, x, y1, x, y2, color=col, lw=2.2)
        ax.text(x + 0.12, 7.05, label, fontsize=9.5, color=col, ha="left", va="center",
                fontweight="bold", linespacing=1.2)

    # --- one node --------------------------------------------------------------
    panel(ax, 0.35, 0.95, 8.3, 5.65, "Camera node k, one of five sites: operator side", TEAL, TEAL_BG)
    row1 = [("CCTV\nvideo", "footage stays\nat the node"),
            ("Detect and\ntrack", "SCRFD and\nByteTrack"),
            ("Quality\ngate", "size, blur,\nexposure, pose"),
            ("Embed", "frozen ArcFace\n+ adapter")]
    row2 = [("Match", "keyed template\nor cosine"),
            ("Track\nvote", "3 of the last\n5 frames"),
            ("Node\nconsole", "operator confirms\nor rejects"),
            ("Local store", "alerts and\naudit trail")]
    for y, row in ((4.15, row1), (2.35, row2)):
        for i, (x, (t, b)) in enumerate(zip(X4, row)):
            box(ax, x, y, W4, H4, t, b, edge=TEAL, size=12)
            if i:
                arrow(ax, X4[i - 1] + W4 + 0.02, y + H4 / 2, x - 0.02, y + H4 / 2, color=TEAL)
    y_mid = (4.15 + 2.35 + H4) / 2
    elbow(ax, [(X4[3] + W4 / 2, 4.13), (X4[3] + W4 / 2, y_mid),
               (X4[0] + W4 / 2, y_mid), (X4[0] + W4 / 2, 2.35 + H4 + 0.02)], color=TEAL)
    box(ax, 0.6, 1.15, X4[3] + W4 - 0.6, 0.95, "Local training",
        "InfoNCE + FedProx on the node's own labelled faces", edge=TEAL, size=12)
    ax.text(4.5, 0.5, "Never leaves a node: raw video, faces of passers-by,\n"
            "training data, labels, pending alerts", ha="center", va="center", fontsize=10.5,
            color=INK, fontweight="bold", linespacing=1.3)
    save(fig, ax, "system_architecture.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (system_architecture, workflow, fl_round, template_flow, node_pipeline,
              fl_architecture):
        f()
        print(f"[figures] wrote {f.__name__}")
