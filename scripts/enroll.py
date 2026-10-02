import argparse
import os
import sys
import time
from PIL import Image, ImageEnhance

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.recognizer import FaceRecognizer
from src.utils import get_image_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enroll student face embeddings from dataset directory."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="project-dataset",
        help="Path to dataset directory containing student roll number folders. Defaults to project-dataset."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings/face_embeddings.pkl",
        help="Path to save face embeddings database (default: embeddings/face_embeddings.pkl)."
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.85,
        help="Minimum MTCNN face detection confidence (default: 0.85)."
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=35,
        help="Minimum face width/height in pixels (default: 35)."
    )
    parser.add_argument(
        "--blur-thresh",
        type=float,
        default=15.0,
        help="Minimum Laplacian variance for blur filter (default: 15.0)."
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


def main():
    args = parse_args()
    dataset_dir = resolve_dataset_path(args.dataset)
    print(f"=== Starting Student Face Enrollment ===")
    print(f"Dataset directory : {dataset_dir}")
    print(f"Output embeddings : {args.output}")

    # Discover student folders
    entries = sorted(os.listdir(dataset_dir))
    student_dirs = [
        d for d in entries
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
    ]

    if not student_dirs:
        print(f"[ERROR] No student folders found in {dataset_dir}!")
        sys.exit(1)

    print(f"Found {len(student_dirs)} students\n")

    # Initialize models
    print("Loading MTCNN detector and FaceNet embedder...")
    start_time = time.time()
    detector = FaceDetector(device=args.device, min_face_size=args.min_size)
    embedder = FaceEmbedder(pretrained="vggface2", device=args.device)
    recognizer = FaceRecognizer()

    total_valid_embeddings = 0
    total_images_processed = 0

    for roll_number in student_dirs:
        student_folder = os.path.join(dataset_dir, roll_number)
        image_files = get_image_paths(student_folder)

        if not image_files:
            print(f"{roll_number} -> 0 images found (skipping)")
            continue

        valid_embeddings = []
        for img_path in image_files:
            total_images_processed += 1
            try:
                img = Image.open(img_path).convert("RGB")
                face_record = detector.detect_primary_face(img, min_confidence=args.min_conf)

                if face_record is not None and face_record["crop"] is not None:
                    crop = face_record["crop"]
                    # 1. Original embedding
                    emb = embedder.embed_crop(crop)
                    valid_embeddings.append(emb)
                    # 2. Flip augmentation: horizontally flipped crop gives a mirrored
                    # viewpoint, stabilizing the centroid and improving pose coverage
                    flipped_crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
                    emb_flip = embedder.embed_crop(flipped_crop)
                    valid_embeddings.append(emb_flip)
                    # 3. Dim lighting simulation (factor 0.85)
                    dim_crop = ImageEnhance.Brightness(crop).enhance(0.85)
                    valid_embeddings.append(embedder.embed_crop(dim_crop))
                    # 4. Bright lighting simulation (factor 1.15)
                    bright_crop = ImageEnhance.Brightness(crop).enhance(1.15)
                    valid_embeddings.append(embedder.embed_crop(bright_crop))
            except Exception as e:
                print(f"  [Warning] Error processing {os.path.basename(img_path)}: {e}")

        if valid_embeddings:
            recognizer.enroll_student(roll_number, valid_embeddings)
            total_valid_embeddings += len(valid_embeddings)
            print(f"{roll_number} -> {len(image_files)} images -> {len(valid_embeddings)} embeddings (flip + lighting augmentation)")
        else:
            print(f"{roll_number} -> {len(image_files)} images -> 0 valid embeddings (Check image quality)")

    # Save database
    recognizer.save_database(args.output)
    elapsed = time.time() - start_time

    print(f"\nEnrollment completed in {elapsed:.2f}s.")
    print(f"Total students   : {len(recognizer.students)}")
    print(f"Total embeddings : {total_valid_embeddings}")
    print(f"Saved to         : {args.output}\n")


if __name__ == "__main__":
    main()
