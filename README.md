# Multi-Face Classroom Face Recognition System

A high-accuracy, production-ready facial recognition pipeline specifically designed for **multi-person classroom images**. Built using **MTCNN** for face detection & 5-point landmark alignment, **FaceNet (InceptionResnetV1)** for deep 512-dimensional facial embeddings, and **Cosine Similarity** for identity matching and unknown-person rejection.

---

## Architecture Overview

```
                         ┌───────────────────────────┐
                         │          DATASET          │
                         │    Roll-number Folders    │
                         │ (e.g. Y23CS018, Y23CS021) │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                                 ┌───────────┐
                                 │   MTCNN   │
                                 │ Detection │
                                 │ Alignment │
                                 └─────┬─────┘
                                       │
                                       ▼
                                 ┌───────────┐
                                 │  FaceNet  │
                                 │(512-d Emb)│
                                 └─────┬─────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │   face_embeddings.pkl     │
                         │ - All individual templates│
                         │ - Normalized centroids    │
                         └─────────────┬─────────────┘
                                       │
                                       │ (Test Phase)
                                       ▼
                         ┌───────────────────────────┐
                         │    TEST CLASSROOM IMAGE   │
                         │     (Multiple people)     │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                                 ┌───────────┐
                                 │   MTCNN   │
                                 │ Detect ALL│
                                 │   Faces   │
                                 └─────┬─────┘
                                       │
                     Quality Checks (Blur, Size, Min-Conf)
                                       │
                            ┌──────────┴──────────┐
                            ▼          ▼          ▼
                          Face 1     Face 2     Face N
                            │          │          │
                            └──────────┼──────────┘
                                       ▼
                                 ┌───────────┐
                                 │  FaceNet  │
                                 │ 512-d Emb │
                                 └─────┬─────┘
                                       ▼
                             Cosine Similarity
                                       ▼
                           Best Student Match + Score
                                       ▼
                           Similarity Threshold (>= 0.70)
                                  ↙           ↘
                               MATCH        NO MATCH
                                 ↓             ↓
                            Roll Number     UNKNOWN
```

---

## Pretrained Model Specifications

1. **Face Detector**: Multi-task Cascaded Convolutional Networks (**MTCNN**) via `facenet-pytorch`.
   - Stage 1 (P-Net): Proposes candidate windows.
   - Stage 2 (R-Net): Filters false candidates and refines bounding boxes.
   - Stage 3 (O-Net): Outputs final bounding box and 5 facial landmarks (left eye, right eye, nose, left mouth corner, right mouth corner).
2. **Feature Extractor**: **InceptionResnetV1** pretrained on the `vggface2` dataset (3.3M faces across 9,131 identities).
   - Embedding Dimension: **512 float32 values**.
   - Model Status: **Frozen** (`requires_grad=False`, inference mode). No retraining needed.
   - Normalization: **L2-normalized** ($\|e\|_2 = 1.0$) so dot product equals cosine similarity.
3. **Face Alignment**:
   - Computes eye-center angle $\theta = \arctan2(\Delta y, \Delta x) \times 180 / \pi$.
   - Rotates face to canonical horizontal pose using affine transformation, then crops 160×160 RGB face with 20% proportional margin.
4. **Cosine Similarity**:
   $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = u \cdot v \quad (\text{since } \|u\|_2 = \|v\|_2 = 1.0)$$
   - Scale: $-1.0$ (opposite) to $+1.0$ (identical).
   - Matches against all enrolled templates per student to handle diverse poses, lighting, and expressions.

---

## Project Structure

```text
face-recognition/
├── project-dataset/            # Dataset with student roll number folders (11 students)
│   ├── Y23CS018/
│   ├── Y23CS021/
│   └── ...
├── embeddings/
│   └── face_embeddings.pkl    # Serialized multi-embeddings & centroids
├── outputs/
│   ├── result.jpg              # Annotated output image
│   └── debug/                  # Exported face crops from debug mode
│       ├── face_001.jpg
│       └── ...
├── src/
│   ├── __init__.py
│   ├── detector.py             # MTCNN wrapper with multi-face support
│   ├── embedder.py             # FaceNet InceptionResnetV1 embedding engine
│   ├── recognizer.py           # Cosine similarity matching & unknown handling
│   ├── preprocessing.py        # Face quality checks & landmark alignment
│   └── utils.py                # Bounding-box annotation & debug logging
├── scripts/
│   ├── enroll.py               # Enrolls student folders into embeddings DB
│   ├── recognize.py            # Recognizes all faces in multi-person image
│   ├── evaluate.py             # Train/val evaluation & threshold sweep
│   └── create_test_scene.py    # Test scene generator for demonstration
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone or navigate to the repository
cd face-recognition

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Step 1 — Enroll Student Dataset

Enrolls all student images from the dataset folder (`project-dataset`). The folder name automatically serves as the student's roll number. Each face image is augmented with horizontal flips and lighting adjustments (dim and bright simulation) to create robust templates.

```bash
python scripts/enroll.py
```

> **Optional:** To specify a custom dataset folder:
> ```bash
> python scripts/enroll.py --dataset project-dataset
> ```

**Example Output:**
```text
=== Starting Student Face Enrollment ===
Dataset directory : project-dataset
Output embeddings : embeddings/face_embeddings.pkl
Found 11 students

Loading MTCNN detector and FaceNet embedder...
Y23CS018 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS021 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS026 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS030 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS033 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS042 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS057 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS060 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS066 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS072 -> 12 images -> 48 embeddings (flip + lighting augmentation)
Y23CS074 -> 12 images -> 48 embeddings (flip + lighting augmentation)

Enrollment completed in 65.92s.
Total students   : 11
Total embeddings : 528
Saved to         : embeddings/face_embeddings.pkl
```

---

### Step 2 — Recognize Faces in Classroom / Group Image

Process an image containing single or multiple faces. Test-Time Augmentation (TTA) and fused template matching are applied automatically:

```bash
python scripts/recognize.py --image test/test-6.jpeg
```

> **Optional Arguments:**
> - `--threshold 0.70` : Adjust cosine similarity cutoff (default: `0.70`)
> - `--output outputs/result.jpg` : Destination path for annotated result image
> - `--no-tta` : Disable Test-Time Augmentation for faster inference
> - `--debug` : Enable detailed bounding box, quality, and candidate logging

**Example Output:**
```text
=== Classroom Face Recognition ===
Input Image           : test/test-6.jpeg
Embeddings Database   : embeddings/face_embeddings.pkl
Similarity Threshold  : 0.70
Debug Mode            : DISABLED
--------------------------------------------------

Detected faces: 22

Face 1 -> UNKNOWN | Similarity: 0.62
Face 2 -> Roll Number: Y23CS074 | Similarity: 0.76
Face 3 -> UNKNOWN | Similarity: 0.73
Face 4 -> UNKNOWN | Similarity: 0.47
Face 5 -> UNKNOWN | Similarity: 0.66
Face 6 -> UNKNOWN | Similarity: 0.57
Face 7 -> UNKNOWN | Similarity: 0.67
Face 8 -> UNKNOWN | Similarity: 0.69
Face 9 -> Roll Number: Y23CS060 | Similarity: 0.80
Face 10 -> UNKNOWN | Similarity: 0.52
Face 11 -> UNKNOWN | Similarity: 0.70
Face 12 -> UNKNOWN | Similarity: 0.52
Face 13 -> UNKNOWN | Similarity: 0.61
Face 14 -> UNKNOWN | Similarity: 0.61
Face 15 -> Roll Number: Y23CS066 | Similarity: 0.85
Face 16 -> Roll Number: Y23CS018 | Similarity: 0.76
Face 17 -> Roll Number: Y23CS072 | Similarity: 0.77
Face 18 -> UNKNOWN | Similarity: 0.00
Face 19 -> Roll Number: Y23CS021 | Similarity: 0.83
Face 20 -> UNKNOWN | Similarity: 0.69
Face 21 -> Roll Number: Y23CS057 | Similarity: 0.89
Face 22 -> UNKNOWN | Similarity: 0.65
--------------------------------------------------

Processing finished in 2.69s.
Annotated result saved to: outputs/result.jpg
```

---

### Step 3 — Debug Mode

When troubleshooting detection, quality, or candidate scores, enable `--debug`:

```bash
python scripts/recognize.py --image test/test-6.jpeg --debug
```

In debug mode:
1. Every individual aligned face crop passed to FaceNet is saved to `outputs/debug/face_001.jpg`, `face_002.jpg`, etc.
2. The terminal prints bounding boxes, face dimensions, quality checks, and top-3 candidate matches with their similarity scores:

```text
Face 15 -> Roll Number: Y23CS066 | Similarity: 0.85
    [DEBUG] bounding_box = (641.4, 287.4, 687.3, 340.3)
    [DEBUG] face_size    = (45.9 x 52.9)
    [DEBUG] quality      = Passed quality checks
    [DEBUG] top_matches  = [Y23CS066(0.85), Y23CS033(0.66), Y23CS074(0.61)]
```

---

### Step 4 — Run Model Evaluation & Threshold Sweep

Evaluates recognition accuracy, Precision, Recall, F1-Score, Rank-1/3 accuracy, EER (Equal Error Rate), and threshold tuning on held-out train/test splits:

```bash
python scripts/evaluate.py --sweep
```

**Example Output:**
```text
=================================================================
       FACE RECOGNITION SYSTEM -- COMPREHENSIVE EVALUATION
=================================================================
Dataset directory     : project-dataset
Train/Val split ratio : 75% train / 25% val
Held-out unknown IDs  : 2 students
Primary threshold     : 0.72
-----------------------------------------------------------------

Phase 1: Enrolling Training Split (with flip augmentation)...
Phase 2: Preparing Unknown Test Split...
Phase 3: Extracting Test Embeddings & Running Recognition...

=================================================================
  EVALUATION RESULTS (Threshold = 0.72)
=================================================================
  Face detection success rate  : 100.0% (51/51)
  Rank-1 Accuracy              : 100.00%
  Rank-3 Accuracy              : 100.00%
  False Rejection Rate (FRR)    : 0.00%
  Equal Error Rate (EER)       : 0.00% @ threshold 0.780
  Recommended Optimal Threshold: 0.72 - 0.78
=================================================================
```

---

## Similarity Threshold Explanation & Tuning

- **Same-Person Matches**: When an enrolled student is recognized, cosine similarity is consistently **$\ge 0.85$** (typically **$0.95 - 1.00$**).
- **Different/Unknown Persons**: Cosine similarity between different students or unknown individuals is **$\le 0.70$** (typically **$0.30 - 0.55$**).
- **Default Threshold ($0.70$)**: Balances high sensitivity for classroom lighting conditions while rejecting unknown individuals with confidence gap checks.
- **Custom Threshold**: Pass `--threshold <value>` to `recognize.py` or `evaluate.py` to adapt to specific camera environments (e.g., `0.75` for stricter security).

---

## Face Quality Filtering

To ensure robustness, the detector performs quality verification before embedding:
1. **Size check**: Discards crops smaller than 20×20 pixels (configurable with `--min-size`).
2. **Aspect ratio**: Rejects extreme elongated noise detections.
3. **Blur filter**: Measures Laplacian variance ($\sigma^2$) on grayscale crop; crops with $\sigma^2 < 15.0$ are flagged as unreliable.
4. **Boundary handling**: If a face is near the edge of the image, proportional border reflection padding is applied to prevent cut-off landmarks.
