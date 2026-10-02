import io
import os
import sys
import time
import base64
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Ensure project root is in sys.path without modifying any existing src files
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.recognizer import FaceRecognizer
from src.utils import draw_annotations, get_image_paths

# Paths
EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "face_embeddings.pkl"
DATASET_PATH = PROJECT_ROOT / "project-dataset"
if not DATASET_PATH.exists():
    DATASET_PATH = PROJECT_ROOT / "dataset"
TEST_DIR = PROJECT_ROOT / "test"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Global model instances (loaded once in memory for ultra-fast inference)
detector: Optional[FaceDetector] = None
embedder: Optional[FaceEmbedder] = None
recognizer: Optional[FaceRecognizer] = None


def sync_dataset_with_embeddings():
    """
    Dynamically scans project-dataset/ folder.
    Automatically enrolls any missing student folders into the recognizer
    and saves the updated database to embeddings/face_embeddings.pkl.
    """
    global detector, embedder, recognizer
    if not DATASET_PATH.exists() or not recognizer:
        return {"total_enrolled": len(recognizer.students) if recognizer else 0, "added": []}

    student_dirs = sorted([
        d.name for d in DATASET_PATH.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    enrolled_set = set(recognizer.students.keys())
    missing_students = [s for s in student_dirs if s not in enrolled_set]

    added = []
    if missing_students and detector and embedder:
        print(f">>> Found {len(missing_students)} new student folder(s) in {DATASET_PATH.name}: {missing_students}")
        for roll_number in missing_students:
            student_folder = DATASET_PATH / roll_number
            img_files = get_image_paths(str(student_folder))
            if not img_files:
                continue

            valid_embeddings = []
            for img_path in img_files:
                try:
                    img = Image.open(img_path).convert("RGB")
                    face_record = detector.detect_primary_face(img, min_confidence=0.85)
                    if face_record is not None and face_record["crop"] is not None:
                        crop = face_record["crop"]
                        # 1. Original
                        valid_embeddings.append(embedder.embed_crop(crop))
                        # 2. Horizontal Flip
                        flipped = crop.transpose(Image.FLIP_LEFT_RIGHT)
                        valid_embeddings.append(embedder.embed_crop(flipped))
                        # 3. Dim lighting simulation
                        dim_crop = ImageEnhance.Brightness(crop).enhance(0.85)
                        valid_embeddings.append(embedder.embed_crop(dim_crop))
                        # 4. Bright lighting simulation
                        bright_crop = ImageEnhance.Brightness(crop).enhance(1.15)
                        valid_embeddings.append(embedder.embed_crop(bright_crop))
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")

            if valid_embeddings:
                recognizer.enroll_student(roll_number, valid_embeddings)
                added.append(roll_number)
                print(f">>> Enrolled {roll_number} with {len(valid_embeddings)} embeddings")

        if added:
            recognizer.save_database(str(EMBEDDINGS_PATH))
            print(f">>> Updated embeddings database saved to {EMBEDDINGS_PATH}")

    return {
        "total_enrolled": len(recognizer.students),
        "students": sorted(list(recognizer.students.keys())),
        "dataset_folders": student_dirs,
        "added": added
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, embedder, recognizer
    print(">>> Initializing AI Models (MTCNN + FaceNet InceptionResnetV1)...")
    detector = FaceDetector(device=None, min_face_size=35)
    embedder = FaceEmbedder(pretrained="vggface2", device=None)
    recognizer = FaceRecognizer(
        embeddings_db_path=str(EMBEDDINGS_PATH) if EMBEDDINGS_PATH.exists() else None,
        similarity_threshold=0.70
    )
    # Automatically scan project-dataset/ and enroll any new students
    sync_dataset_with_embeddings()
    enrolled_count = len(recognizer.students) if recognizer else 0
    print(f">>> Ready! Enrolled students: {enrolled_count}")
    yield


app = FastAPI(
    title="Face Recognition Attendance & Roll Number System",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
@app.get("/api/health")
def health_check():
    """Standard health check endpoint for cloud platforms."""
    return {"status": "ok", "ready": recognizer is not None}


def process_image_recognition(image: Image.Image, threshold: float = 0.70):
    """
    Executes detection, embedding, and recognition pipeline on a PIL Image.
    Returns structured results and base64-encoded annotated image.
    """
    t0 = time.time()
    
    # 1. MTCNN Face Detection & Alignment
    faces = detector.detect_all_faces(
        image=image,
        min_confidence=0.85,
        min_size=35,
        align=True
    )

    detected_count = len(faces)
    detections_with_results = []
    recognized_roll_numbers_set = set()
    recognized_faces = []

    for face in faces:
        face_idx = face["index"]
        box = face["box"]
        crop = face["crop"]
        is_valid = face["is_valid"]
        quality_reason = face.get("quality_reason", "")

        if not is_valid and crop is None:
            identity = "UNKNOWN"
            similarity = 0.0
            rec_result = {
                "identity": identity,
                "similarity": similarity,
                "top_candidates": []
            }
        else:
            emb = embedder.embed_crop(crop, tta=True)
            rec_result = recognizer.recognize_single(emb, threshold=threshold)
            identity = rec_result["identity"]
            similarity = rec_result["similarity"]

        is_match = identity != "UNKNOWN" and rec_result.get("is_match", False)

        if is_match:
            recognized_roll_numbers_set.add(identity)

        record = {
            "index": face_idx,
            "box": [float(coord) for coord in box],
            "identity": identity,
            "similarity": float(similarity),
            "is_valid": is_valid,
            "quality_reason": quality_reason,
            "rec_result": rec_result,
            "crop": crop
        }
        detections_with_results.append(record)

        recognized_faces.append({
            "face_index": face_idx,
            "roll_number": identity,
            "is_recognized": is_match,
            "similarity": round(float(similarity), 3),
            "similarity_percent": int(round(float(similarity) * 100)),
            "box": [round(float(coord), 1) for coord in box],
            "is_valid": is_valid,
            "quality_reason": quality_reason
        })

    # Draw visual annotations on canvas
    annotated_pil = draw_annotations(
        image=image,
        detections=detections_with_results,
        output_path=None
    )

    # Encode annotated image to JPEG base64 for direct browser rendering
    buffer = io.BytesIO()
    annotated_pil.save(buffer, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    annotated_data_url = f"data:image/jpeg;base64,{img_b64}"

    elapsed = round(time.time() - t0, 2)
    sorted_roll_numbers = sorted(list(recognized_roll_numbers_set))

    return {
        "success": True,
        "total_faces_detected": detected_count,
        "recognized_count": len(sorted_roll_numbers),
        "unknown_count": detected_count - len(sorted_roll_numbers),
        "roll_numbers": sorted_roll_numbers,
        "processing_time_seconds": elapsed,
        "faces": recognized_faces,
        "annotated_image": annotated_data_url
    }


@app.get("/api/status")
def get_status():
    dataset_students = []
    if DATASET_PATH.exists():
        dataset_students = sorted([
            d.name for d in DATASET_PATH.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    enrolled_students = sorted(list(recognizer.students.keys())) if recognizer else []
    return {
        "status": "ready" if recognizer else "loading",
        "enrolled_count": len(enrolled_students),
        "dataset_count": len(dataset_students),
        "enrolled_students": enrolled_students,
        "dataset_students": dataset_students,
        "default_threshold": recognizer.threshold if recognizer else 0.70
    }


@app.post("/api/sync")
def sync_dataset_endpoint():
    """Trigger manual re-scan of project-dataset/ folder to enroll new people."""
    result = sync_dataset_with_embeddings()
    return {"success": True, **result}


@app.get("/api/samples")
def list_sample_images():
    """List available sample test images for quick one-click testing."""
    if not TEST_DIR.exists():
        return {"samples": []}
    
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    samples = []
    for file in sorted(TEST_DIR.iterdir()):
        if file.suffix.lower() in valid_exts:
            samples.append({
                "filename": file.name,
                "size_kb": round(file.stat().st_size / 1024, 1),
            })
    return {"samples": samples}


@app.get("/api/sample/{filename}")
def get_sample_image(filename: str):
    """Serve a sample image from the test directory."""
    safe_filename = Path(filename).name
    file_path = TEST_DIR / safe_filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(file_path)


@app.post("/api/recognize-sample")
def recognize_sample(filename: str = Query(...), threshold: float = Query(0.70)):
    """Run recognition directly on an existing sample test image."""
    safe_filename = Path(filename).name
    file_path = TEST_DIR / safe_filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    
    try:
        image = Image.open(file_path).convert("RGB")
        result = process_image_recognition(image, threshold=threshold)
        result["filename"] = safe_filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recognize")
async def recognize_upload(
    file: UploadFile = File(...),
    threshold: float = Query(0.70)
):
    """
    Primary endpoint: Upload an image file and retrieve all detected roll numbers.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = process_image_recognition(image, threshold=threshold)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")


# Mount frontend static directory
if not FRONTEND_DIR.exists():
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Face Recognition Frontend</h1><p>Initializing...</p>")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    print(f"Starting server on http://{host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)
