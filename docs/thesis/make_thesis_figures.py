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


def box(ax, x, y, w, h, title, body="", edge=TEAL, face="white", size=9.0, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.6, edgecolor=edge, facecolor=face))
    if body:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=INK, wrap=True)
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center",
                fontsize=size - 2, color=MUTED, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.3)


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


def chain(ax, xs, y, w, h, items, edge, face="white"):
    for i, (x, (t, b)) in enumerate(zip(xs, items)):
        box(ax, x, y, w, h, t, b, edge=edge, face=face)
        if i:
            arrow(ax, xs[i - 1] + w + 0.02, y + h / 2, x - 0.02, y + h / 2)


def workflow():
    fig, ax = canvas(12, 6.6)
    panel(ax, 0.1, 3.45, 11.8, 3.05, "Training phase  (dataset preparation on AWS GPU, federated training)", TEAL, TEAL_BG)
    panel(ax, 0.1, 0.1, 11.8, 3.05, "Operational phase  (consoles and camera video)", BROWN, BROWN_BG)
    w, h = 1.82, 1.15
    xs = [0.25 + i * 1.95 for i in range(6)]
    chain(ax, xs, 4.85, w, h, [
        ("VGGFace2", "8,631 identities\n172,620 images"),
        ("Align", "SCRFD landmarks\n112x112 ArcFace crop"),
        ("Degrade", "one camera profile\nper node"),
        ("Embed", "frozen ArcFace\nclean + degraded"),
        ("Partition", "70% test identities\n30% over 5 nodes"),
        ("Federated training", "FedProx, 30 rounds\nearly stopping"),
    ], TEAL)
    box(ax, xs[5], 3.62, w, 0.85, "adapter_final.pt", "66,240 parameters", edge=TEAL)
    arrow(ax, xs[5] + w / 2, 4.83, xs[5] + w / 2, 4.5)
    xs2 = [0.25 + i * 1.95 for i in range(6)]
    chain(ax, xs2, 1.55, w, h, [
        ("Open case", "photo read once;\nthumbnail kept"),
        ("Keyed templates", "one 64-byte code\nper node"),
        ("Scan video", "detect, track,\nquality gate"),
        ("Embed + match", "ArcFace + adapter\ncosine / template"),
        ("Track vote", "3 of last 5 frames\n-> pending alert"),
        ("Human review", "confirm / reject\nat the node"),
    ], BROWN)
    box(ax, xs2[4], 0.3, w, 0.85, "Audit log", "every action recorded", edge=BROWN)
    box(ax, xs2[5], 0.3, w, 0.85, "Confirmed sighting", "one crop to server", edge=BROWN)
    arrow(ax, xs2[5] + w / 2, 1.53, xs2[5] + w / 2, 1.17)
    arrow(ax, xs2[5] + 0.05, 1.53, xs2[4] + w - 0.1, 1.17)
    arrow(ax, xs[5] + 0.3, 3.6, xs2[3] + w - 0.25, 2.73, color=TEAL)
    ax.text(xs[5] - 0.12, 4.05, "trained adapter,\nused by every node", fontsize=8.5,
            color=TEAL, ha="right", va="center", linespacing=1.25)
    fig.savefig(OUT / "workflow.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fl_round():
    fig, ax = canvas(12, 5.8)
    panel(ax, 2.2, 3.7, 7.6, 1.95, "Server", BROWN, BROWN_BG)
    box(ax, 2.5, 3.85, 3.4, 1.2, "Aggregate",
        "w_g <- w_g + sum_k (n_k / n) * delta_k\npairwise masks cancel in the sum", edge=BROWN)
    box(ax, 6.15, 3.85, 3.4, 1.2, "Validate and keep best round",
        "rank-1 on 518 held-out identities\n(disjoint from training and test)", edge=BROWN)
    arrow(ax, 5.92, 4.45, 6.13, 4.45)
    ax.add_patch(FancyBboxPatch((0.1, 0.1), 11.8, 2.05,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=2.0, edgecolor=TEAL, facecolor=TEAL_BG))
    ax.text(6.0, 0.3, "Five camera nodes: images, embeddings and labels never leave a node",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEAL)
    for k in range(5):
        x = 0.35 + k * 2.33
        box(ax, x, 0.55, 2.0, 1.4, f"Node {k}",
            "local training\nInfoNCE + FedProx\n1 epoch", edge=TEAL)
        arrow(ax, x + 0.55, 3.68, x + 0.55, 1.97, color=BROWN)
        arrow(ax, x + 1.45, 1.97, x + 1.45, 3.68, color=TEAL)
    # legend, beside the server
    arrow(ax, 0.25, 5.3, 0.25, 4.7, color=BROWN)
    ax.text(0.4, 5.0, "global adapter w_g\n(0.26 MB) to\nevery node", fontsize=8.5,
            color=BROWN, ha="left", va="center")
    arrow(ax, 10.0, 4.7, 10.0, 5.3, color=TEAL)
    ax.text(10.15, 5.0, "delta_k only\n+ optional DP\n  clip & noise\n+ optional\n  pairwise mask",
            fontsize=8.5, color=TEAL, ha="left", va="center")
    fig.savefig(OUT / "fl_round.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def template_flow():
    fig, ax = canvas(12, 4.9)
    w, h = 1.95, 1.2
    xs = [0.2, 2.4, 4.6, 6.8]
    chain(ax, xs, 3.0, w, h, [
        ("Reference photo", "read once, in memory"),
        ("ArcFace embedding", "512 floats"),
        ("Binarise", "each dimension vs\nits mean -> 512 bits"),
        ("XOR with key_k", "key_k = SHA-256(master,\ncase, node k)"),
    ], BROWN)
    for k in range(5):
        y = 3.95 - k * 0.72
        box(ax, 9.25, y - 0.27, 2.55, 0.56, f"node {k}: 64-byte template", edge=BROWN, size=8.5,
            bold=False)
        arrow(ax, 8.77, 3.6, 9.23, y, color=BROWN)
    ax.text(9.25, 4.55, "different bit pattern at every node", fontsize=8.5, color=BROWN)
    chain(ax, [2.4, 4.6, 6.8], 0.35, w, h, [
        ("Live face at node k", "ArcFace + adapter"),
        ("Binarise + XOR key_k", "same key as the template"),
        ("Hamming similarity", "s = 1 - 2 d_H / 512\nalert if s >= 0.2656"),
    ], TEAL)
    arrow(ax, 9.23, 0.96, 6.8 + w + 0.02, 0.95, color=TEAL)
    ax.text(9.3, 0.6, "compared without undoing the key", fontsize=8.5, color=TEAL)
    fig.savefig(OUT / "template_flow.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def node_pipeline():
    fig, ax = canvas(12.6, 2.6)
    w, h = 1.42, 1.75
    items = [
        ("Frames", "sampled at\n<= 8 fps"),
        ("Detect", "SCRFD det_10g\n5 landmarks"),
        ("Track", "ByteTrack\none track\nper person"),
        ("Quality gate", "eyes >= 24 px\nblur, exposure\nyaw <= 45 deg"),
        ("Embed", "aligned crop\nArcFace +\nadapter"),
        ("Match", "cosine >= 0.24\nor template\n>= 0.2656"),
        ("Vote", "3 of last 5\nframes of the\ntrack"),
        ("Pending alert", "crop + score\nfor the node's\noperator"),
    ]
    xs = [0.15 + i * 1.56 for i in range(len(items))]
    chain(ax, xs, 0.45, w, h, items, TEAL)
    fig.savefig(OUT / "node_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (workflow, fl_round, template_flow, node_pipeline):
        f()
        print(f"[figures] wrote {f.__name__}")
