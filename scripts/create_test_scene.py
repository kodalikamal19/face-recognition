import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def create_classroom_composite():
    os.makedirs("test", exist_ok=True)
    
    from src.utils import get_image_paths

    student_ids = ["Y23CS018", "Y23CS021", "Y23CS026", "Y23CS030", "Y23CS042", "Y23CS066", "Y23CS072"]
    people = []
    
    # Add 5 enrolled students
    for sid in student_ids[:5]:
        imgs = get_image_paths(f"dataset/{sid}")
        if imgs:
            people.append((imgs[-1], sid))
            
    # Add 1 unknown person
    unknown_portrait = r"C:\Users\kodal\.gemini\antigravity-ide\brain\a80f7a23-807d-4228-841b-919424b915f0\unknown_student_face_1788796093171.jpg"
    people.insert(3, (unknown_portrait, "UNKNOWN_PERSON"))
    
    # 2 rows, 3 columns layout on a subtle textured classroom background
    cell_w = 400
    cell_h = 500
    cols = 3
    rows = 2
    margin_x = 40
    margin_y = 40
    gap = 30
    
    total_w = margin_x * 2 + cols * cell_w + (cols - 1) * gap
    total_h = margin_y * 2 + rows * cell_h + (rows - 1) * gap
    
    # Background
    bg = np.full((total_h, total_w, 3), 235, dtype=np.uint8)
    
    # Add subtle classroom backdrop gradient
    for y in range(total_h):
        shade = int(220 + 25 * (y / total_h))
        bg[y, :, :] = [shade - 10, shade - 5, shade]
    
    for idx, (img_path, label) in enumerate(people):
        r = idx // cols
        c = idx % cols
        
        x = margin_x + c * (cell_w + gap)
        y = margin_y + r * (cell_h + gap)
        
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            # Resize image to fit nicely in cell while maintaining aspect ratio
            img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            
            img_np = np.array(img)
            ih, iw = img_np.shape[:2]
            
            # Center within cell
            ox = x + (cell_w - iw) // 2
            oy = y + (cell_h - ih) // 2
            
            bg[oy:oy+ih, ox:ox+iw] = img_np
    
    out_path = os.path.join("test", "classroom.jpg")
    Image.fromarray(bg).save(out_path, quality=95)
    print(f"Created multi-face test scene at: {out_path} ({total_w}x{total_h})")

if __name__ == "__main__":
    create_classroom_composite()
