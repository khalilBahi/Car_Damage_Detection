from PIL import Image
import os
from pathlib import Path

base_dir = Path('/home/khalil_b/test_pfe/data')
target_size = (224, 224)

for split in ['train', 'val', 'test']:
    split_dir = base_dir / split
    for cls in os.listdir(split_dir):
        cls_dir = split_dir / cls
        for img_name in os.listdir(cls_dir):
            img_path = cls_dir / img_name
            if img_path.is_file() and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(img_path) as img:
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)  # Better interpolation
                    img_resized.save(img_path)
print("Image resizing complete.")