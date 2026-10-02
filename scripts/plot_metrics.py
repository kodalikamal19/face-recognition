"""
plot_metrics.py  -  Face Recognition System | Visual Evaluation Dashboard
==========================================================================
Runs the full evaluation pipeline (same logic as evaluate.py) and produces
a publication-quality multi-panel figure saved to:

    outputs/evaluation_report.png

Panels:
  1. Confusion Matrix  (normalised heat-map)
  2. Threshold Sweep   (Accuracy / FAR / FRR / F1 vs threshold)
  3. DET Curve         (FAR vs FRR with EER marked)
  4. Per-Class Metrics (Precision / Recall / F1 grouped bar chart)
  5. Summary Card      (key numbers at a glance)

Usage:
    python scripts/plot_metrics.py [--dataset <path>] [--threshold 0.70]
                                   [--train-ratio 0.75] [--heldout-unknowns 2]
                                   [--device cuda|cpu] [--output <path>]
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageEnhance

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.recognizer import FaceRecognizer
from src.utils import get_image_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate visual accuracy report for the Face Recognition pipeline."
    )
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--heldout-unknowns", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default="outputs/evaluation_report.png")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def resolve_dataset(path):
    if path and os.path.isdir(path):
        return path
    for c in ("project-dataset", "dataset"):
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("Could not find dataset. Specify one with --dataset <path>.")


def compute_eer(far_list, frr_list, thresholds):
    best_eer, best_t = 1.0, thresholds[0]
    for i in range(len(thresholds) - 1):
        if far_list[i] >= frr_list[i] and far_list[i + 1] <= frr_list[i + 1]:
            fd = far_list[i] - far_list[i + 1]
            rd = frr_list[i + 1] - frr_list[i]
            tot = fd + rd
            if tot > 0:
                alpha = (far_list[i] - frr_list[i]) / tot
                eer = far_list[i] - alpha * fd
                t   = thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i])
            else:
                eer = (far_list[i] + frr_list[i]) / 2
                t   = thresholds[i]
            if abs(eer) < abs(best_eer):
                best_eer, best_t = eer, t
    if best_eer >= 1.0:
        diffs = [abs(f - r) for f, r in zip(far_list, frr_list)]
        idx = diffs.index(min(diffs))
        best_eer = (far_list[idx] + frr_list[idx]) / 2
        best_t   = thresholds[idx]
    return max(0.0, best_eer), best_t


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def run_evaluation(args):
    dataset_dir = resolve_dataset(args.dataset)
    student_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ])
    if len(student_dirs) < 3:
        sys.exit("[ERROR] Need at least 3 students to evaluate unknown rejection.")

    num_unknowns  = min(args.heldout_unknowns, max(1, len(student_dirs) // 4))
    enrolled_stds = student_dirs[:-num_unknowns]
    unknown_stds  = student_dirs[-num_unknowns:]
    all_labels    = sorted(enrolled_stds + ["UNKNOWN"])

    print(f"Dataset          : {dataset_dir}")
    print(f"Enrolled students: {len(enrolled_stds)}  |  Unknown: {len(unknown_stds)}")
    print(f"Primary threshold: {args.threshold}\n")

    detector   = FaceDetector(device=args.device)
    embedder   = FaceEmbedder(pretrained="vggface2", device=args.device)
    recognizer = FaceRecognizer()

    # Phase 1: Enrol
    print("Phase 1: Enrolling training split ...")
    test_samples       = []
    detection_failures = 0

    for roll in enrolled_stds:
        folder = os.path.join(dataset_dir, roll)
        images = get_image_paths(folder)
        split  = max(1, int(round(len(images) * args.train_ratio)))
        train_imgs, val_imgs = images[:split], images[split:]

        valid_embs = []
        for img_p in train_imgs:
            try:
                img  = Image.open(img_p).convert("RGB")
                face = detector.detect_primary_face(img)
                if face and face["crop"] is not None:
                    crop = face["crop"]
                    valid_embs.append(embedder.embed_crop(crop))
                    valid_embs.append(embedder.embed_crop(crop.transpose(Image.FLIP_LEFT_RIGHT)))
                    valid_embs.append(embedder.embed_crop(ImageEnhance.Brightness(crop).enhance(0.85)))
                    valid_embs.append(embedder.embed_crop(ImageEnhance.Brightness(crop).enhance(1.15)))
                else:
                    detection_failures += 1
            except Exception:
                detection_failures += 1

        if valid_embs:
            recognizer.enroll_student(roll, valid_embs)

        for img_p in val_imgs:
            test_samples.append({"path": img_p, "ground_truth": roll, "is_unknown": False})

    # Phase 2: Unknown pool
    print("Phase 2: Preparing unknown test split ...")
    for roll in unknown_stds:
        folder = os.path.join(dataset_dir, roll)
        for img_p in get_image_paths(folder):
            test_samples.append({
                "path": img_p, "ground_truth": "UNKNOWN",
                "original_identity": roll, "is_unknown": True
            })

    # Phase 3: Extract test embeddings
    print("Phase 3: Extracting test embeddings ...")
    processed = []
    for sample in test_samples:
        try:
            img  = Image.open(sample["path"]).convert("RGB")
            face = detector.detect_primary_face(img)
            if face and face["crop"] is not None:
                sample["embedding"] = embedder.embed_crop(face["crop"], tta=True)
                sample["detected"]  = True
                processed.append(sample)
            else:
                detection_failures += 1
        except Exception:
            detection_failures += 1

    det_rate = len(processed) / len(test_samples) * 100 if test_samples else 0.0
    print(f"Detection success : {det_rate:.1f}%  ({len(processed)}/{len(test_samples)})\n")

    # Evaluation function
    def evaluate_at(thresh):
        ck = ik = rk = cu = au = 0
        tot_k = tot_u = 0
        r1c = r3c = r_tot = 0
        per_tp = defaultdict(int)
        per_fp = defaultdict(int)
        per_fn = defaultdict(int)
        confusion = defaultdict(lambda: defaultdict(int))

        for sample in processed:
            rec  = recognizer.recognize_single(sample["embedding"], threshold=thresh)
            pred = rec["identity"]
            gt   = sample["ground_truth"]
            tops = rec.get("top_candidates", [])

            confusion[gt][pred] += 1

            if not sample["is_unknown"]:
                tot_k += 1; r_tot += 1
                if tops and tops[0]["roll_number"] == gt:
                    r1c += 1
                if gt in [c["roll_number"] for c in tops[:3]]:
                    r3c += 1
                if pred == gt:
                    ck += 1; per_tp[gt] += 1
                elif pred == "UNKNOWN":
                    rk += 1; per_fn[gt] += 1
                else:
                    ik += 1; per_fn[gt] += 1; per_fp[pred] += 1
            else:
                tot_u += 1
                if pred == "UNKNOWN":
                    cu += 1; per_tp["UNKNOWN"] += 1
                else:
                    au += 1; per_fn["UNKNOWN"] += 1; per_fp[pred] += 1

        tot = tot_k + tot_u
        acc = (ck + cu) / tot       if tot   > 0 else 0.0
        far = au / tot_u             if tot_u > 0 else 0.0
        frr = rk / tot_k             if tot_k > 0 else 0.0
        r1  = r1c / r_tot            if r_tot > 0 else 0.0
        r3  = r3c / r_tot            if r_tot > 0 else 0.0

        pcm = {}
        for lbl in all_labels:
            tp = per_tp[lbl]; fp = per_fp[lbl]; fn = per_fn[lbl]
            pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
            pcm[lbl] = {"precision": pr, "recall": rc, "f1": f1,
                         "tp": tp, "fp": fp, "fn": fn}

        valid = [m for m in pcm.values() if (m["tp"] + m["fn"]) > 0]
        mac_p = sum(m["precision"] for m in valid) / len(valid) if valid else 0.0
        mac_r = sum(m["recall"]    for m in valid) / len(valid) if valid else 0.0
        mac_f = sum(m["f1"]        for m in valid) / len(valid) if valid else 0.0

        return {
            "threshold": thresh, "accuracy": acc,
            "far": far, "frr": frr,
            "macro_precision": mac_p, "macro_recall": mac_r, "macro_f1": mac_f,
            "rank1": r1, "rank3": r3,
            "per_class": pcm, "confusion": dict(confusion),
            "correct_known": ck, "incorrect_known": ik, "rejected_known": rk,
            "correct_unknown": cu, "accepted_unknown": au,
            "total_known": tot_k, "total_unknown": tot_u,
        }

    print(f"Evaluating at primary threshold {args.threshold} ...")
    primary = evaluate_at(args.threshold)

    print("Running threshold sweep (0.40 to 0.89) ...")
    sweep_thresholds = [round(t * 0.01, 2) for t in range(40, 90)]
    sweep_results    = [evaluate_at(t) for t in sweep_thresholds]

    far_list = [r["far"]      for r in sweep_results]
    frr_list = [r["frr"]      for r in sweep_results]
    acc_list = [r["accuracy"] for r in sweep_results]
    f1_list  = [r["macro_f1"] for r in sweep_results]

    eer_val, eer_t = compute_eer(far_list, frr_list, sweep_thresholds)

    combined    = [r["accuracy"] - 0.5 * r["far"] - 0.5 * r["frr"] for r in sweep_results]
    best_thresh = sweep_thresholds[combined.index(max(combined))]

    print(f"EER              : {eer_val*100:.2f}% @ threshold {eer_t:.3f}")
    print(f"Best threshold   : {best_thresh:.2f}\n")

    return {
        "primary": primary,
        "sweep_thresholds": sweep_thresholds,
        "far_list": far_list, "frr_list": frr_list,
        "acc_list": acc_list, "f1_list":  f1_list,
        "eer_val": eer_val, "eer_t": eer_t,
        "best_thresh": best_thresh,
        "all_labels": all_labels,
        "det_rate": det_rate,
        "total_test": len(processed),
        "enrolled_count": len(enrolled_stds),
        "unknown_count":  len(unknown_stds),
        "dataset_dir": dataset_dir,
    }


# ---------------------------------------------------------------------------
# Colour palette (dark-mode)
# ---------------------------------------------------------------------------
BG     = "#0f1117"
PANEL  = "#1a1d27"
BORDER = "#2a2d3a"
TEXT   = "#e8eaf0"
ACCENT = "#6c63ff"
GREEN  = "#22d3a5"
RED    = "#ff6b6b"
YELLOW = "#ffd166"
BLUE   = "#4ecdc4"
ORANGE = "#ff9f43"
PURPLE = "#a29bfe"
MUTED  = "#8891a8"

SWEEP_COLORS = {
    "Accuracy": GREEN,
    "FAR":      RED,
    "FRR":      YELLOW,
    "Macro F1": BLUE,
}


def style_axes(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10, color=TEXT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(color=BORDER, linestyle="--", linewidth=0.5, alpha=0.6)


# ---------------------------------------------------------------------------
# Panel 1: Confusion Matrix
# ---------------------------------------------------------------------------
def plot_confusion(ax, data):
    labels = data["all_labels"]
    conf   = data["primary"]["confusion"]
    n      = len(labels)
    mat    = np.zeros((n, n), dtype=int)
    for i, tl in enumerate(labels):
        for j, pl in enumerate(labels):
            mat[i, j] = conf.get(tl, {}).get(pl, 0)

    row_sums = mat.sum(axis=1, keepdims=True)
    norm_mat = np.where(row_sums > 0, mat / row_sums, 0.0)

    cmap = LinearSegmentedColormap.from_list(
        "dark_blue", [PANEL, "#2d2f7e", ACCENT, "#00e5ff"], N=256
    )
    im = ax.imshow(norm_mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    fs = max(5, 8 - n // 5)
    for i in range(n):
        for j in range(n):
            nrm = norm_mat[i, j]
            color = "white" if nrm < 0.6 else "#0f1117"
            ax.text(j, i, f"{mat[i,j]}\n({nrm*100:.0f}%)",
                    ha="center", va="center", fontsize=fs,
                    color=color, fontweight="bold")

    short = [lb[:10] for lb in labels]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(short, fontsize=fs)
    ax.set_xlabel("Predicted",  fontsize=8, color=TEXT)
    ax.set_ylabel("True Label", fontsize=8, color=TEXT)
    ax.set_title("Confusion Matrix (row-normalised)",
                 fontsize=10, fontweight="bold", color=TEXT, pad=10)
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=TEXT)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT, labelsize=7)
    cbar.set_label("Recall fraction", color=TEXT, fontsize=7)


# ---------------------------------------------------------------------------
# Panel 2: Threshold Sweep
# ---------------------------------------------------------------------------
def plot_sweep(ax, data):
    thresholds = data["sweep_thresholds"]
    series = {
        "Accuracy": data["acc_list"],
        "FAR":      data["far_list"],
        "FRR":      data["frr_list"],
        "Macro F1": data["f1_list"],
    }
    for name, vals in series.items():
        lw = 2.0 if name in ("Accuracy", "Macro F1") else 1.5
        ls = "-"  if name in ("Accuracy", "Macro F1") else "--"
        ax.plot(thresholds, [v * 100 for v in vals],
                color=SWEEP_COLORS[name], linewidth=lw, linestyle=ls,
                label=name, zorder=3)

    pt = data["primary"]["threshold"]
    ax.axvline(pt,                color=ACCENT,  linewidth=1.2, linestyle=":", alpha=0.85,
               label=f"Primary ({pt:.2f})")
    ax.axvline(data["eer_t"],     color=ORANGE,  linewidth=1.0, linestyle=":", alpha=0.70,
               label=f"EER ({data['eer_t']:.2f})")
    ax.axvline(data["best_thresh"], color=GREEN, linewidth=1.2, linestyle=":", alpha=0.70,
               label=f"Best ({data['best_thresh']:.2f})")

    style_axes(ax, "Threshold Sweep", "Threshold", "Rate (%)")
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
              loc="upper right", framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel 3: DET Curve (FAR vs FRR)
# ---------------------------------------------------------------------------
def plot_roc(ax, data):
    far = data["far_list"]
    frr = data["frr_list"]
    thresholds = data["sweep_thresholds"]

    for i in range(len(far) - 1):
        ax.plot([far[i] * 100, far[i + 1] * 100],
                [frr[i] * 100, frr[i + 1] * 100],
                color=ACCENT, linewidth=2.0, alpha=0.85)

    lim = [0, 100]
    ax.plot(lim, lim, color=MUTED, linewidth=0.8, linestyle="--",
            alpha=0.5, label="FAR = FRR")

    eer_pct = data["eer_val"] * 100
    ax.scatter([eer_pct], [eer_pct], color=ORANGE, s=80, zorder=5,
               label=f"EER = {eer_pct:.1f}%")

    pt_idx = min(range(len(thresholds)),
                 key=lambda i: abs(thresholds[i] - data["primary"]["threshold"]))
    ax.scatter([far[pt_idx] * 100], [frr[pt_idx] * 100],
               color=GREEN, s=80, zorder=5, marker="^",
               label=f"Primary t={data['primary']['threshold']:.2f}")

    style_axes(ax, "FAR vs FRR (DET Curve)", "FAR (%)", "FRR (%)")
    ax.set_xlim(-1, min(100, max(far) * 100 + 5))
    ax.set_ylim(-1, min(100, max(frr) * 100 + 5))
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
              framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel 4: Per-class bar chart
# ---------------------------------------------------------------------------
def plot_per_class(ax, data):
    labels = data["all_labels"]
    pcm    = data["primary"]["per_class"]
    n      = len(labels)
    x      = np.arange(n)
    w      = 0.25

    prec = [pcm.get(l, {}).get("precision", 0) * 100 for l in labels]
    rec  = [pcm.get(l, {}).get("recall",    0) * 100 for l in labels]
    f1   = [pcm.get(l, {}).get("f1",        0) * 100 for l in labels]

    bp = ax.bar(x - w, prec, w, label="Precision", color=BLUE,   alpha=0.88, zorder=3)
    br = ax.bar(x,      rec,  w, label="Recall",    color=GREEN,  alpha=0.88, zorder=3)
    bf = ax.bar(x + w,  f1,   w, label="F1-Score",  color=PURPLE, alpha=0.88, zorder=3)

    fs = max(4, 7 - n // 5)
    if n <= 20:
        for bars, vals in [(bp, prec), (br, rec), (bf, f1)]:
            for rect, val in zip(bars, vals):
                if val > 2:
                    ax.text(rect.get_x() + rect.get_width() / 2,
                            rect.get_height() + 1,
                            f"{val:.0f}", ha="center", va="bottom",
                            fontsize=fs, color=TEXT)

    short = [lb[:8] for lb in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=fs)
    style_axes(ax, "Per-Class Precision / Recall / F1", "Student ID", "Score (%)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
              loc="upper right", framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel 5: Summary card
# ---------------------------------------------------------------------------
def plot_summary(ax, data):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)

    pr  = data["primary"]
    det = data["det_rate"]

    metrics = [
        ("Overall Accuracy",       f"{pr['accuracy']*100:.2f}%",        GREEN),
        ("Detection Rate",         f"{det:.1f}%",                        BLUE),
        ("Macro Precision",        f"{pr['macro_precision']*100:.2f}%",  ACCENT),
        ("Macro Recall",           f"{pr['macro_recall']*100:.2f}%",     ACCENT),
        ("Macro F1-Score",         f"{pr['macro_f1']*100:.2f}%",         ACCENT),
        ("False Acceptance (FAR)", f"{pr['far']*100:.2f}%",              RED),
        ("False Rejection (FRR)",  f"{pr['frr']*100:.2f}%",              YELLOW),
        ("Rank-1 Accuracy",        f"{pr['rank1']*100:.2f}%",            GREEN),
        ("Rank-3 Accuracy",        f"{pr['rank3']*100:.2f}%",            GREEN),
        ("EER",                    f"{data['eer_val']*100:.2f}% @ {data['eer_t']:.3f}", ORANGE),
        ("Primary Threshold",      f"{pr['threshold']:.2f}",             PURPLE),
        ("Best Threshold",         f"{data['best_thresh']:.2f}",         PURPLE),
        ("Enrolled Students",      str(data["enrolled_count"]),           TEXT),
        ("Unknown Students",       str(data["unknown_count"]),            TEXT),
        ("Test Faces Evaluated",   str(data["total_test"]),               TEXT),
    ]

    title_y = 0.97
    ax.text(0.5, title_y, "Evaluation Summary",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold", color=TEXT)

    row_h = (title_y - 0.08) / len(metrics)
    for idx, (name, value, color) in enumerate(metrics):
        y = title_y - 0.07 - idx * row_h
        if idx % 2 == 0:
            rect = mpatches.FancyBboxPatch(
                (0.01, y - row_h * 0.5), 0.98, row_h,
                boxstyle="round,pad=0.005", linewidth=0,
                facecolor=BORDER, transform=ax.transAxes, alpha=0.35
            )
            ax.add_patch(rect)
        ax.text(0.05, y, name + ":", transform=ax.transAxes,
                ha="left", va="center", fontsize=8, color=MUTED)
        ax.text(0.95, y, value, transform=ax.transAxes,
                ha="right", va="center", fontsize=8.5,
                fontweight="bold", color=color)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data = run_evaluation(args)

    plt.rcParams.update({
        "figure.facecolor": BG,
        "text.color":       TEXT,
        "font.family":      "DejaVu Sans",
        "font.size":        9,
    })

    fig = plt.figure(figsize=(22, 16), facecolor=BG)
    fig.suptitle(
        "Face Recognition System  -  Comprehensive Accuracy Report",
        fontsize=16, fontweight="bold", color=TEXT, y=0.98
    )

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        left=0.05, right=0.98,
        top=0.95, bottom=0.06,
        hspace=0.45, wspace=0.35,
        width_ratios=[1.6, 1.2, 0.85],
        height_ratios=[1.3, 1, 1]
    )

    ax_cm       = fig.add_subplot(gs[0, 0])
    ax_sweep    = fig.add_subplot(gs[1, 0])
    ax_roc      = fig.add_subplot(gs[2, 0])
    ax_perclass = fig.add_subplot(gs[:2, 1])
    ax_summary  = fig.add_subplot(gs[:, 2])

    plot_confusion(ax_cm,        data)
    plot_sweep(ax_sweep,         data)
    plot_roc(ax_roc,             data)
    plot_per_class(ax_perclass,  data)
    plot_summary(ax_summary,     data)

    footer = (
        f"Dataset: {data['dataset_dir']}   |   "
        f"Model: InceptionResnetV1 (VGGFace2)   |   "
        f"Threshold: {data['primary']['threshold']:.2f}   |   "
        f"Best: {data['best_thresh']:.2f}   |   "
        f"EER: {data['eer_val']*100:.2f}%"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom",
             fontsize=7.5, color=MUTED, style="italic")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print(f"Report saved -> {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
