import os
import shutil
from pathlib import Path
import random

source_dir = Path('../data/processed')
train_dir = Path('../data/train')
val_dir = Path('../data/val')
test_dir = Path('../data/test')

for dir_path in [train_dir, val_dir, test_dir]:
    dir_path.mkdir(exist_ok=True)
    for cls in os.listdir(source_dir):
        (dir_path / cls).mkdir(exist_ok=True)

for cls in os.listdir(source_dir):
    cls_path = source_dir / cls
    images = os.listdir(cls_path)
    random.shuffle(images)
    total = len(images)
    train_count = int(0.8 * total)
    val_count = int(0.1 * total)

    for img in images[:train_count]:
        shutil.move(cls_path / img, train_dir / cls / img)
    for img in images[train_count:train_count + val_count]:
        shutil.move(cls_path / img, val_dir / cls / img)
    for img in images[train_count + val_count:]:
        shutil.move(cls_path / img, test_dir / cls / img)

print("Data split complete. Check counts:", {cls: len(os.listdir(train_dir / cls)) for cls in os.listdir(source_dir)})