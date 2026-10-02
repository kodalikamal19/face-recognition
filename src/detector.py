from typing import Any
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

from src.preprocessing import align_and_crop_face, check_face_quality


class FaceDetector:
    """
    MTCNN-based Face Detection and Alignment engine.
    Detects single or multiple faces in an image and aligns them with 5 facial landmarks.
    """

    def __init__(
        self,
        device: torch.device | str | None = None,
        min_face_size: int = 35,
        thresholds: list[float] | None = None,
        factor: float = 0.709,
        post_process: bool = False
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]

        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=min_face_size,
            thresholds=thresholds,
            factor=factor,
            post_process=post_process,
            keep_all=True,
            device=self.device
        )

    def detect_all_faces(
        self,
        image: Image.Image | np.ndarray,
        min_confidence: float = 0.85,
        min_size: int = 35,
        blur_threshold: float = 15.0,
        align: bool = True
    ) -> list[dict[str, Any]]:
        """
        Detect all faces in an image, perform quality checks, and align face crops.

        Args:
            image: Input image (PIL Image or RGB numpy array).
            min_confidence: Minimum detection confidence to accept.
            min_size: Minimum width/height in pixels.
            blur_threshold: Minimum Laplacian variance for blur filter.
            align: Whether to align rotationally using landmarks.

        Returns:
            List of detected face dictionaries:
            [
                {
                    "box": [x1, y1, x2, y2],
                    "prob": float,
                    "landmarks": np.ndarray,
                    "crop": PIL.Image,
                    "is_valid": bool,
                    "quality_reason": str,
                    "blur_score": float
                },
                ...
            ]
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype("uint8")).convert("RGB")
            np_img = image
        else:
            pil_img = image.convert("RGB")
            np_img = np.array(pil_img)

        boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

        if boxes is None or len(boxes) == 0:
            return []

        results = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            p = float(prob) if prob is not None else 0.0
            lm = landmarks[i] if landmarks is not None else None

            is_valid, reason, blur = check_face_quality(
                image_np=np_img,
                box=box,
                prob=p,
                min_size=min_size,
                blur_threshold=blur_threshold,
                min_conf=min_confidence
            )

            aligned_crop = None
            if is_valid or p >= min_confidence:
                aligned_crop = align_and_crop_face(
                    image=pil_img,
                    box=box,
                    landmarks=lm if align else None,
                    output_size=160,
                    margin_ratio=0.20
                )

            results.append({
                "index": i + 1,
                "box": [float(b) for b in box],
                "prob": p,
                "landmarks": lm,
                "crop": aligned_crop,
                "is_valid": is_valid,
                "quality_reason": reason,
                "blur_score": blur
            })

        return results

    def detect_primary_face(
        self,
        image: Image.Image | np.ndarray,
        min_confidence: float = 0.85
    ) -> dict[str, Any] | None:
        """
        For enrollment images: detect and select the primary (largest/most confident) face.
        Filters out spurious background detections.
        """
        faces = self.detect_all_faces(image, min_confidence=min_confidence)
        valid_faces = [f for f in faces if f["is_valid"] and f["crop"] is not None]

        if not valid_faces:
            # Fallback to any detected face with sufficient confidence
            fallback = [f for f in faces if f["crop"] is not None and f["prob"] >= min_confidence]
            if not fallback:
                return None
            valid_faces = fallback

        # Sort by area * confidence
        def face_score(f: dict[str, Any]) -> float:
            box = f["box"]
            area = (box[2] - box[0]) * (box[3] - box[1])
            return area * f["prob"]

        return max(valid_faces, key=face_score)
