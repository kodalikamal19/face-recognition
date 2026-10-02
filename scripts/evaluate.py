import argparse
import os
import sys
import time
import numpy as np
from PIL import Image, ImageEnhance
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.recognizer import FaceRecognizer
from src.utils import get_image_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Face Recognition Pipeline on Held-Out Test Data"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset directory. Defaults to project-dataset or dataset."
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Proportion of images per student used for enrollment (default: 0.75 = 9 of 12 images)."
    )
    parser.add_argument(
        "--heldout-unknowns",
        type=int,
        default=2,
        help="Number of student identities to completely hold out as 'UNKNOWN' persons (default: 2)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Primary similarity threshold to evaluate (default: 0.70)."
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Perform threshold sweep from 0.40 to 0.85 to find optimal threshold."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu')."
    )
    return parser.parse_args()


def resolve_dataset_path(provided_path: str | None) -> str:
    if provided_path and os.path.isdir(provided_path):
        return provided_path
    if os.path.isdir("project-dataset"):
        return "project-dataset"
    if os.path.isdir("dataset"):
        return "dataset"
    raise FileNotFoundError(
        "Could not find dataset directory. Please specify with --dataset <path>"
    )


def compute_eer(far_values: list[float], frr_values: list[float], thresholds: list[float]) -> tuple[float, float]:
    """
    Compute Equal Error Rate (EER) — the threshold where FAR ≈ FRR.
    This is a standard biometric evaluation metric.

    Returns:
        (eer_value, eer_threshold)
    """
    if len(far_values) < 2:
        return 0.0, thresholds[0] if thresholds else 0.0

    best_eer = 1.0
    best_thresh = thresholds[0]

    for i in range(len(thresholds) - 1):
        # FAR decreases and FRR increases as threshold increases
        # Find where they cross
        if far_values[i] >= frr_values[i] and far_values[i + 1] <= frr_values[i + 1]:
            # Linear interpolation to find crossing point
            far_diff = far_values[i] - far_values[i + 1]
            frr_diff = frr_values[i + 1] - frr_values[i]
            total_diff = far_diff + frr_diff
            if total_diff > 0:
                alpha = (far_values[i] - frr_values[i]) / total_diff
                eer = far_values[i] - alpha * far_diff
                thresh = thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i])
            else:
                eer = (far_values[i] + frr_values[i]) / 2
                thresh = thresholds[i]

            if abs(eer) < abs(best_eer):
                best_eer = eer
                best_thresh = thresh

    # Fallback: find threshold with smallest |FAR - FRR|
    if best_eer >= 1.0:
        diffs = [abs(f - r) for f, r in zip(far_values, frr_values)]
        min_idx = diffs.index(min(diffs))
        best_eer = (far_values[min_idx] + frr_values[min_idx]) / 2
        best_thresh = thresholds[min_idx]

    return max(0.0, best_eer), best_thresh


def main():
    args = parse_args()
    dataset_dir = resolve_dataset_path(args.dataset)

    print("=" * 65)
    print("       FACE RECOGNITION SYSTEM -- COMPREHENSIVE EVALUATION")
    print("=" * 65)
    print(f"Dataset directory     : {dataset_dir}")
    print(f"Train/Val split ratio : {args.train_ratio * 100:.0f}% train / {(1 - args.train_ratio) * 100:.0f}% val")
    print(f"Held-out unknown IDs  : {args.heldout_unknowns} students")
    print(f"Primary threshold     : {args.threshold}")
    print("-" * 65)

    # Discover student folders
    student_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ])

    if len(student_dirs) < 3:
        print("[ERROR] Need at least 3 students to evaluate held-out unknown rejection.")
        sys.exit(1)

    # Split students into Enrolled vs Held-Out Unknown
    num_unknowns = min(args.heldout_unknowns, max(1, len(student_dirs) // 4))
    enrolled_students = student_dirs[:-num_unknowns]
    unknown_students = student_dirs[-num_unknowns:]

    print(f"Enrolled Students ({len(enrolled_students)})   : {', '.join(enrolled_students)}")
    print(f"Unknown Students ({len(unknown_students)})    : {', '.join(unknown_students)}")
    print("-" * 65)

    # Initialize models
    detector = FaceDetector(device=args.device)
    embedder = FaceEmbedder(pretrained="vggface2", device=args.device)
    recognizer = FaceRecognizer()

    # Step 1: Enroll training split of known students (with flip augmentation)
    print("\nPhase 1: Enrolling Training Split (with flip augmentation)...")
    test_samples: list[dict] = []
    total_images = 0
    detection_failures = 0

    for roll in enrolled_students:
        folder = os.path.join(dataset_dir, roll)
        images = get_image_paths(folder)
        total_images += len(images)
        split_idx = max(1, int(round(len(images) * args.train_ratio)))

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        valid_embs = []
        for img_p in train_imgs:
            try:
                img = Image.open(img_p).convert("RGB")
                face = detector.detect_primary_face(img)
                if face and face["crop"] is not None:
                    crop = face["crop"]
                    # 1. Original embedding
                    emb = embedder.embed_crop(crop)
                    valid_embs.append(emb)
                    # 2. Flip augmentation
                    flipped = crop.transpose(Image.FLIP_LEFT_RIGHT)
                    emb_flip = embedder.embed_crop(flipped)
                    valid_embs.append(emb_flip)
                    # 3. Dim lighting simulation (0.85)
                    dim_crop = ImageEnhance.Brightness(crop).enhance(0.85)
                    valid_embs.append(embedder.embed_crop(dim_crop))
                    # 4. Bright lighting simulation (1.15)
                    bright_crop = ImageEnhance.Brightness(crop).enhance(1.15)
                    valid_embs.append(embedder.embed_crop(bright_crop))
                else:
                    detection_failures += 1
            except Exception:
                detection_failures += 1

        if valid_embs:
            recognizer.enroll_student(roll, valid_embs)
            print(f"  [Enroll] {roll} -> {len(valid_embs)} embeddings enrolled (flip + lighting)")

        # Add val images to test set
        for img_p in val_imgs:
            test_samples.append({
                "path": img_p,
                "ground_truth": roll,
                "is_unknown": False
            })

    # Step 2: Add all images of held-out unknown students to test set
    print("\nPhase 2: Preparing Unknown Test Split...")
    for roll in unknown_students:
        folder = os.path.join(dataset_dir, roll)
        images = get_image_paths(folder)
        total_images += len(images)
        for img_p in images:
            test_samples.append({
                "path": img_p,
                "ground_truth": "UNKNOWN",
                "original_identity": roll,
                "is_unknown": True
            })
        print(f"  [Unknown] {roll} -> {len(images)} images designated as UNKNOWN tests")

    num_known_tests = sum(1 for s in test_samples if not s['is_unknown'])
    num_unknown_tests = sum(1 for s in test_samples if s['is_unknown'])
    print(f"\nTotal test queries: {len(test_samples)} ({num_known_tests} known, {num_unknown_tests} unknown)")

    # Step 3: Extract test embeddings
    print("\nPhase 3: Extracting Test Embeddings & Running Recognition...")
    processed_test_data = []
    for sample in test_samples:
        try:
            img = Image.open(sample["path"]).convert("RGB")
            face = detector.detect_primary_face(img)
            if face and face["crop"] is not None:
                emb = embedder.embed_crop(face["crop"], tta=True)
                sample["embedding"] = emb
                sample["detected"] = True
                processed_test_data.append(sample)
            else:
                detection_failures += 1
                sample["detected"] = False
        except Exception:
            detection_failures += 1

    # ─────────────────────────────────────────────────────────
    # Evaluation function for a given threshold
    # ─────────────────────────────────────────────────────────
    all_labels = sorted(enrolled_students + ["UNKNOWN"])

    def evaluate_at_threshold(thresh: float):
        correct_known = 0
        incorrect_known = 0
        rejected_known = 0  # False Rejection (FRR)
        correct_unknown = 0  # True Negative
        accepted_unknown = 0  # False Acceptance (FAR)

        total_known_tests = 0
        total_unknown_tests = 0

        # Per-class tracking for precision/recall
        per_class_tp = defaultdict(int)
        per_class_fp = defaultdict(int)
        per_class_fn = defaultdict(int)

        # Rank tracking
        rank1_correct = 0
        rank3_correct = 0
        rank_total_known = 0

        # Confusion matrix: confusion[true_label][predicted_label] = count
        confusion = defaultdict(lambda: defaultdict(int))

        predictions = []

        for sample in processed_test_data:
            rec = recognizer.recognize_single(sample["embedding"], threshold=thresh)
            predicted = rec["identity"]
            gt = sample["ground_truth"]
            top_candidates = rec.get("top_candidates", [])

            predictions.append({
                "ground_truth": gt,
                "predicted": predicted,
                "similarity": rec["similarity"],
                "top_candidates": top_candidates
            })

            confusion[gt][predicted] += 1

            if not sample["is_unknown"]:
                total_known_tests += 1
                rank_total_known += 1

                # Rank-1: is the correct identity the top candidate?
                if top_candidates and top_candidates[0]["roll_number"] == gt:
                    rank1_correct += 1

                # Rank-3: is the correct identity in top 3?
                top3_rolls = [c["roll_number"] for c in top_candidates[:3]]
                if gt in top3_rolls:
                    rank3_correct += 1

                if predicted == gt:
                    correct_known += 1
                    per_class_tp[gt] += 1
                elif predicted == "UNKNOWN":
                    rejected_known += 1
                    per_class_fn[gt] += 1
                else:
                    incorrect_known += 1
                    per_class_fn[gt] += 1
                    per_class_fp[predicted] += 1
            else:
                total_unknown_tests += 1
                if predicted == "UNKNOWN":
                    correct_unknown += 1
                    per_class_tp["UNKNOWN"] += 1
                else:
                    accepted_unknown += 1
                    per_class_fn["UNKNOWN"] += 1
                    per_class_fp[predicted] += 1

        total_evaluated = total_known_tests + total_unknown_tests
        total_correct = correct_known + correct_unknown
        acc = (total_correct / total_evaluated) if total_evaluated > 0 else 0.0
        far = (accepted_unknown / total_unknown_tests) if total_unknown_tests > 0 else 0.0
        frr = (rejected_known / total_known_tests) if total_known_tests > 0 else 0.0

        # Compute per-class precision, recall, F1
        per_class_metrics = {}
        for label in all_labels:
            tp = per_class_tp[label]
            fp = per_class_fp[label]
            fn = per_class_fn[label]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class_metrics[label] = {
                "precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn
            }

        # Macro-averaged metrics
        valid_classes = [m for m in per_class_metrics.values() if (m["tp"] + m["fn"]) > 0]
        macro_precision = sum(m["precision"] for m in valid_classes) / len(valid_classes) if valid_classes else 0.0
        macro_recall = sum(m["recall"] for m in valid_classes) / len(valid_classes) if valid_classes else 0.0
        macro_f1 = sum(m["f1"] for m in valid_classes) / len(valid_classes) if valid_classes else 0.0

        # Rank accuracies
        rank1_acc = (rank1_correct / rank_total_known) if rank_total_known > 0 else 0.0
        rank3_acc = (rank3_correct / rank_total_known) if rank_total_known > 0 else 0.0

        return {
            "threshold": thresh,
            "total_test_faces": total_evaluated,
            "correct_known": correct_known,
            "incorrect_known": incorrect_known,
            "rejected_known": rejected_known,
            "correct_unknown": correct_unknown,
            "accepted_unknown": accepted_unknown,
            "accuracy": acc,
            "far": far,
            "frr": frr,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class_metrics": per_class_metrics,
            "rank1_accuracy": rank1_acc,
            "rank3_accuracy": rank3_acc,
            "confusion": dict(confusion),
            "predictions": predictions
        }

    # ---------------------------------------------------------
    # Primary Evaluation at user threshold
    # ---------------------------------------------------------
    start_eval = time.time()
    res = evaluate_at_threshold(args.threshold)
    eval_time = time.time() - start_eval
    det_rate = (len(processed_test_data) / len(test_samples) * 100) if test_samples else 0.0

    print("\n" + "=" * 65)
    print(f"  EVALUATION RESULTS (Threshold = {args.threshold:.2f})")
    print("=" * 65)

    # Core metrics
    print(f"\n{'-'*40}")
    print(f"  CORE METRICS")
    print(f"{'-'*40}")
    print(f"  Face detection success rate  : {det_rate:.1f}% ({len(processed_test_data)}/{len(test_samples)})")
    print(f"  Total test faces evaluated   : {res['total_test_faces']}")
    print(f"  Correctly recognized (known) : {res['correct_known']}")
    print(f"  Incorrectly recognized       : {res['incorrect_known']}")
    print(f"  Unknown correctly rejected   : {res['correct_unknown']}")
    print(f"  Unknown falsely accepted     : {res['accepted_unknown']} (FAR)")
    print(f"  Known falsely rejected       : {res['rejected_known']} (FRR)")

    print(f"\n{'-'*40}")
    print(f"  ACCURACY & ERROR RATES")
    print(f"{'-'*40}")
    print(f"  Overall Accuracy             : {res['accuracy'] * 100:.2f}%")
    print(f"  False Acceptance Rate (FAR)   : {res['far'] * 100:.2f}%")
    print(f"  False Rejection Rate (FRR)    : {res['frr'] * 100:.2f}%")

    print(f"\n{'-'*40}")
    print(f"  PRECISION / RECALL / F1 (Macro-Avg)")
    print(f"{'-'*40}")
    print(f"  Macro Precision              : {res['macro_precision'] * 100:.2f}%")
    print(f"  Macro Recall                 : {res['macro_recall'] * 100:.2f}%")
    print(f"  Macro F1-Score               : {res['macro_f1'] * 100:.2f}%")

    print(f"\n{'-'*40}")
    print(f"  RANK ACCURACY (Known Faces)")
    print(f"{'-'*40}")
    print(f"  Rank-1 Accuracy              : {res['rank1_accuracy'] * 100:.2f}%")
    print(f"  Rank-3 Accuracy              : {res['rank3_accuracy'] * 100:.2f}%")

    # Per-class breakdown
    print(f"\n{'-'*40}")
    print(f"  PER-CLASS BREAKDOWN")
    print(f"{'-'*40}")
    print(f"  {'Class':<12} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'TP':>3} | {'FP':>3} | {'FN':>3}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*3}-+-{'-'*3}-+-{'-'*3}")
    for label in all_labels:
        m = res["per_class_metrics"].get(label, {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0})
        print(f"  {label:<12} | {m['precision']*100:>5.1f}% | {m['recall']*100:>5.1f}% | {m['f1']*100:>5.1f}% | {m['tp']:>3} | {m['fp']:>3} | {m['fn']:>3}")

    # Confusion Matrix
    print(f"\n{'-'*40}")
    print(f"  CONFUSION MATRIX")
    print(f"{'-'*40}")
    cm_labels = sorted(set(
        list(res["confusion"].keys()) +
        [p for row in res["confusion"].values() for p in row.keys()]
    ))
    # Header
    header = f"  {'True \\ Pred':<12}"
    for pl in cm_labels:
        short = pl[:8] if len(pl) > 8 else pl
        header += f" | {short:>8}"
    print(header)
    print(f"  {'-' * (13 + 11 * len(cm_labels))}")
    # Rows
    for tl in cm_labels:
        row_str = f"  {tl:<12}"
        for pl in cm_labels:
            count = res["confusion"].get(tl, {}).get(pl, 0)
            row_str += f" | {count:>8}"
        print(row_str)

    # ---------------------------------------------------------
    # Threshold Sweep + EER
    # ---------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  THRESHOLD SWEEP ANALYSIS")
    print(f"{'='*65}")
    print(f"  {'Thresh':<7} | {'Acc':>7} | {'FAR':>7} | {'FRR':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'R-1':>6} | Note")
    print(f"  {'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+------")

    best_thresh = args.threshold
    best_score = -1.0

    sweep_thresholds = [round(t * 0.01, 2) for t in range(40, 90)]  # 0.40 to 0.89 in 0.01 steps
    far_list = []
    frr_list = []

    for t in sweep_thresholds:
        s_res = evaluate_at_threshold(t)
        far_list.append(s_res["far"])
        frr_list.append(s_res["frr"])

        # Combined score: maximize accuracy while penalizing FAR & FRR
        combined = s_res["accuracy"] - (0.5 * s_res["far"] + 0.5 * s_res["frr"])
        if combined > best_score:
            best_score = combined
            best_thresh = t

        # Only print at 0.05 intervals for readability
        if abs(t * 100 - round(t * 100 / 5) * 5) < 0.01:
            tag = " <== Current" if abs(t - args.threshold) < 1e-4 else ""
            if abs(t - best_thresh) < 1e-4 and not tag:
                tag = " <== Best"
            print(
                f"  {t:<7.2f} | {s_res['accuracy'] * 100:>6.1f}% | {s_res['far'] * 100:>6.1f}% | "
                f"{s_res['frr'] * 100:>6.1f}% | {s_res['macro_precision'] * 100:>6.1f}% | "
                f"{s_res['macro_recall'] * 100:>6.1f}% | {s_res['macro_f1'] * 100:>6.1f}% | "
                f"{s_res['rank1_accuracy'] * 100:>5.1f}% |{tag}"
            )

    # Compute EER
    eer_value, eer_threshold = compute_eer(far_list, frr_list, sweep_thresholds)

    print(f"  {'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+------")
    print(f"\n  Equal Error Rate (EER)       : {eer_value * 100:.2f}% @ threshold {eer_threshold:.3f}")
    print(f"  Recommended Optimal Threshold: {best_thresh:.2f}")
    print(f"  Evaluation time              : {eval_time:.2f}s")
    print(f"\n{'='*65}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

