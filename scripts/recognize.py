import argparse
import os
import sys
import time
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.recognizer import FaceRecognizer
from src.utils import draw_annotations, save_debug_crops


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Face Classroom Image Recognition Pipeline"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the classroom/multi-person test image."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="embeddings/face_embeddings.pkl",
        help="Path to the enrolled embeddings pickle file (default: embeddings/face_embeddings.pkl)."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold for unknown rejection (default: 0.70)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.jpg",
        help="Path to save the annotated output image (default: outputs/result.jpg)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: logs coordinates/sizes and exports face crops to outputs/debug/."
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.85,
        help="Minimum MTCNN detection confidence (default: 0.85)."
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=35,
        help="Minimum face width/height in pixels (default: 35)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu')."
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable Test-Time Augmentation (TTA) during embedding extraction."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"[ERROR] Test image not found at '{args.image}'")
        sys.exit(1)

    if not os.path.isfile(args.embeddings):
        print(f"[ERROR] Enrolled embeddings file not found at '{args.embeddings}'.")
        print("Please run enrollment first: python scripts/enroll.py")
        sys.exit(1)

    print(f"=== Classroom Face Recognition ===")
    print(f"Input Image           : {args.image}")
    print(f"Embeddings Database   : {args.embeddings}")
    print(f"Similarity Threshold  : {args.threshold}")
    print(f"Debug Mode            : {'ENABLED' if args.debug else 'DISABLED'}")
    print("-" * 50)

    # Load Image
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to open image: {e}")
        sys.exit(1)

    # Load models
    t0 = time.time()
    detector = FaceDetector(device=args.device, min_face_size=args.min_size)
    embedder = FaceEmbedder(pretrained="vggface2", device=args.device)
    recognizer = FaceRecognizer(
        embeddings_db_path=args.embeddings,
        similarity_threshold=args.threshold
    )

    # 1. Detection & Alignment
    faces = detector.detect_all_faces(
        image=image,
        min_confidence=args.min_conf,
        min_size=args.min_size,
        align=True
    )

    detected_count = len(faces)
    print(f"\nDetected faces: {detected_count}\n")

    if detected_count == 0:
        print("No faces detected in the provided image.")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        image.save(args.output)
        print(f"Saved unmodified image to: {args.output}")
        return

    # Debug crops export
    if args.debug:
        debug_dir = os.path.join("outputs", "debug")
        saved_crops = save_debug_crops(debug_dir, faces)
        print(f"[DEBUG] Saved {len(saved_crops)} aligned face crops to '{debug_dir}/'\n")

    # 2. Embedding & Recognition
    detections_with_results = []
    for face in faces:
        face_idx = face["index"]
        box = face["box"]
        crop = face["crop"]
        is_valid = face["is_valid"]
        quality_reason = face["quality_reason"]
        w = box[2] - box[0]
        h = box[3] - box[1]

        if not is_valid and crop is None:
            # Face failed critical checks and could not be cropped
            identity = "UNKNOWN"
            similarity = 0.0
            rec_result = {
                "identity": identity,
                "similarity": similarity,
                "top_candidates": []
            }
        else:
            emb = embedder.embed_crop(crop, tta=not args.no_tta)
            rec_result = recognizer.recognize_single(emb, threshold=args.threshold)
            identity = rec_result["identity"]
            similarity = rec_result["similarity"]

        record = {
            "index": face_idx,
            "box": box,
            "identity": identity,
            "similarity": similarity,
            "is_valid": is_valid,
            "rec_result": rec_result,
            "crop": crop
        }
        detections_with_results.append(record)

        # Standard Terminal Output
        if identity == "UNKNOWN":
            print(f"Face {face_idx} -> UNKNOWN | Similarity: {similarity:.2f}")
        else:
            print(f"Face {face_idx} -> Roll Number: {identity} | Similarity: {similarity:.2f}")

        # Debug Detailed Terminal Output
        if args.debug:
            top_str = ", ".join(
                f"{c['roll_number']}({c['similarity']:.2f})"
                for c in rec_result.get("top_candidates", [])
            )
            print(f"    [DEBUG] bounding_box = ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
            print(f"    [DEBUG] face_size    = ({w:.1f} x {h:.1f})")
            print(f"    [DEBUG] quality      = {quality_reason}")
            print(f"    [DEBUG] top_matches  = [{top_str}]")
            print()

    # 3. Save Annotated Output Image
    print("-" * 50)
    annotated_img = draw_annotations(
        image=image,
        detections=detections_with_results,
        output_path=args.output
    )
    total_time = time.time() - t0
    print(f"\nProcessing finished in {total_time:.2f}s.")
    print(f"Annotated result saved to: {args.output}\n")


if __name__ == "__main__":
    main()
