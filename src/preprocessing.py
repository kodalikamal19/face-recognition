import cv2
import numpy as np
from PIL import Image
import torch


def check_face_quality(
    image_np: np.ndarray,
    box: np.ndarray,
    prob: float = 1.0,
    min_size: int = 35,
    blur_threshold: float = 20.0,
    min_conf: float = 0.85
) -> tuple[bool, str, float]:
    """
    Perform quality checks on a detected face candidate.

    Args:
        image_np: Source image in RGB format (H, W, 3).
        box: Bounding box [x1, y1, x2, y2].
        prob: Detection confidence score from MTCNN.
        min_size: Minimum width and height required for reliable recognition.
        blur_threshold: Minimum Laplacian variance threshold to detect blurriness.
        min_conf: Minimum MTCNN detection confidence.

    Returns:
        (is_valid, reason, blur_score)
    """
    if prob < min_conf:
        return False, f"Low detection confidence ({prob:.2f} < {min_conf:.2f})", 0.0

    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    if w < min_size or h < min_size:
        return False, f"Face too small ({int(w)}x{int(h)} < {min_size}px)", 0.0

    # Aspect ratio check (abnormal elongated crops)
    aspect_ratio = h / max(1.0, w)
    if aspect_ratio < 0.5 or aspect_ratio > 2.5:
        return False, f"Abnormal aspect ratio ({aspect_ratio:.2f})", 0.0

    # Safe crop bounds for blur check
    img_h, img_w = image_np.shape[:2]
    ix1 = max(0, min(img_w - 1, int(round(x1))))
    iy1 = max(0, min(img_h - 1, int(round(y1))))
    ix2 = max(0, min(img_w, int(round(x2))))
    iy2 = max(0, min(img_h, int(round(y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return False, "Invalid face crop coordinates", 0.0

    crop = image_np[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return False, "Empty face crop", 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if blur_score < blur_threshold:
        return False, f"Face too blurry (Laplacian var {blur_score:.1f} < {blur_threshold:.1f})", blur_score

    return True, "Passed quality checks", blur_score


def align_and_crop_face(
    image: Image.Image | np.ndarray,
    box: np.ndarray | list[float],
    landmarks: np.ndarray | None = None,
    output_size: int = 160,
    margin_ratio: float = 0.20
) -> Image.Image:
    """
    Align face rotationally using eye landmarks (if available) and crop to standard FaceNet input size.

    Args:
        image: PIL Image or RGB numpy array.
        box: [x1, y1, x2, y2].
        landmarks: 5 facial landmarks [[x_le, y_le], [x_re, y_re], [x_nose, y_nose], ...].
        output_size: Target square dimension (default 160 for FaceNet).
        margin_ratio: Margin to expand bounding box around face (e.g. 0.20 = 20%).

    Returns:
        Aligned and cropped PIL Image of shape (output_size, output_size, 3).
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()

    img_h, img_w = img_np.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    w = x2 - x1
    h = y2 - y1

    # Attempt landmark-based rotational alignment if landmarks for both eyes exist
    if landmarks is not None and len(landmarks) >= 2:
        left_eye = np.array(landmarks[0], dtype=np.float32)
        right_eye = np.array(landmarks[1], dtype=np.float32)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # Only rotate if the angle is within a reasonable tilt range (-45 to 45 deg)
        if abs(angle) <= 45.0:
            eye_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
            rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            rotated_img = cv2.warpAffine(
                img_np,
                rot_matrix,
                (img_w, img_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            img_np = rotated_img

            # Transform face center to rotated space
            face_center = np.array([x1 + w / 2.0, y1 + h / 2.0, 1.0])
            new_cx, new_cy = rot_matrix.dot(face_center)
            x1 = new_cx - w / 2.0
            y1 = new_cy - h / 2.0
            x2 = new_cx + w / 2.0
            y2 = new_cy + h / 2.0

    # Expand box by margin
    margin_w = w * margin_ratio
    margin_h = h * margin_ratio
    crop_x1 = int(round(x1 - margin_w))
    crop_y1 = int(round(y1 - margin_h))
    crop_x2 = int(round(x2 + margin_w))
    crop_y2 = int(round(y2 + margin_h))

    # Calculate padding if crop goes outside image boundaries
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - img_w)
    pad_bottom = max(0, crop_y2 - img_h)

    # Clamped bounds
    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(img_w, crop_x2)
    src_y2 = min(img_h, crop_y2)

    crop = img_np[src_y1:src_y2, src_x1:src_x2]

    # Apply padding if necessary
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101
        )

    # Resize to canonical output_size (160x160) — BICUBIC preserves sharper facial details
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(resized)


def apply_clahe(image_pil: Image.Image, clip_limit: float = 2.0, grid_size: int = 8) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize
    lighting variations across different capture conditions.

    Operates on the L channel of LAB color space to preserve color information.

    Args:
        image_pil: Input PIL Image (RGB).
        clip_limit: CLAHE contrast clipping limit.
        grid_size: Size of the local region grid.

    Returns:
        CLAHE-enhanced PIL Image (RGB).
    """
    img_np = np.array(image_pil.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_enhanced)


def fixed_image_standardization(tensor: torch.Tensor) -> torch.Tensor:
    """
    Standardize image tensor according to FaceNet requirements: (x - 127.5) / 128.0
    """
    return (tensor - 127.5) / 128.0
