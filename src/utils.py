import os
from typing import Any
import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_paths(folder_path: str) -> list[str]:
    """Return all valid image paths in a folder sorted alphabetically."""
    if not os.path.isdir(folder_path):
        return []
    valid_paths = []
    for entry in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            _, ext = os.path.splitext(entry)
            if ext.lower() in IMAGE_EXTENSIONS:
                valid_paths.append(full_path)
    return valid_paths


def save_debug_crops(
    output_dir: str,
    faces_data: list[dict[str, Any]]
) -> list[str]:
    """
    Save individual aligned face crops to disk for visual debugging.

    Args:
        output_dir: Target directory (e.g. outputs/debug/).
        faces_data: List of face records with 'crop' PIL image.

    Returns:
        List of saved image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for i, face in enumerate(faces_data):
        crop = face.get("crop")
        idx = face.get("index", i + 1)
        filename = f"face_{idx:03d}.jpg"
        filepath = os.path.join(output_dir, filename)

        if crop is not None:
            if isinstance(crop, Image.Image):
                crop.save(filepath, quality=95)
            elif isinstance(crop, np.ndarray):
                bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, bgr)
            saved_paths.append(filepath)

    return saved_paths


def draw_annotations(
    image: Image.Image | np.ndarray,
    detections: list[dict[str, Any]],
    output_path: str | None = None
) -> Image.Image:
    """
    Draw bounding boxes, roll numbers, and similarity scores on an image.

    Args:
        image: Original test image.
        detections: List of detection records with 'box', 'identity', 'similarity', 'is_valid'.
        output_path: Optional path to save the annotated image.

    Returns:
        Annotated PIL Image.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image.copy()

    # Convert RGB to BGR for OpenCV drawing
    canvas = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_h, img_w = canvas.shape[:2]

    # Dynamic line thickness and font scale based on image dimensions
    base_dim = max(img_w, img_h)
    line_thickness = max(2, int(round(base_dim / 500)))
    font_scale = max(0.5, base_dim / 1400)

    for i, det in enumerate(detections):
        box = det["box"]
        x1, y1, x2, y2 = [int(round(coord)) for coord in box]
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        identity = det.get("identity", "UNKNOWN")
        sim = det.get("similarity", 0.0)
        is_valid = det.get("is_valid", True)
        face_idx = det.get("index", i + 1)

        # Color scheme (BGR):
        # Green for recognized match
        # Red/Orange for UNKNOWN
        # Amber/Yellow for low quality
        if not is_valid:
            box_color = (0, 165, 255)      # Amber / Orange
            label = f"#{face_idx} UNRELIABLE ({sim:.2f})"
        elif identity == "UNKNOWN":
            box_color = (60, 76, 231)      # Red / Crimson
            label = f"#{face_idx} UNKNOWN ({sim:.2f})"
        else:
            box_color = (80, 180, 46)      # Fresh Green
            label = f"#{face_idx} {identity} ({sim:.2f})"

        # Draw bounding rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, line_thickness)

        # Draw decorative corner brackets for sleek visual appearance
        corner_len = max(10, int((x2 - x1) * 0.15))
        corner_thickness = line_thickness + 1
        # Top-left
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), box_color, corner_thickness)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), box_color, corner_thickness)
        # Top-right
        cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), box_color, corner_thickness)
        cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), box_color, corner_thickness)
        # Bottom-left
        cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), box_color, corner_thickness)
        cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), box_color, corner_thickness)
        # Bottom-right
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), box_color, corner_thickness)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), box_color, corner_thickness)

        # Draw label background banner
        font = cv2.FONT_HERSHEY_DUPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        banner_h = text_h + baseline + 10
        banner_w = text_w + 14

        # Position banner above the box if there is room, otherwise inside/below
        if y1 - banner_h >= 0:
            by1 = y1 - banner_h
            by2 = y1
        else:
            by1 = y1
            by2 = y1 + banner_h

        bx1 = x1
        bx2 = min(img_w, x1 + banner_w)

        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), box_color, -1)
        cv2.putText(
            canvas,
            label,
            (bx1 + 7, by2 - baseline - 4),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Convert back to RGB PIL Image
    annotated_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        annotated_pil.save(output_path, quality=95)

    return annotated_pil
