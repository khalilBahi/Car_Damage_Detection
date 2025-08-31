import csv
import shutil
from pathlib import Path
import sys

# Use absolute paths
raw_dir = Path('/home/khalil_b/test_pfe/data/raw/CarDD_release/CarDD_COCO')
processed_dir = Path('/home/khalil_b/test_pfe/data/processed')
processed_dir.mkdir(exist_ok=True)

# Six target classes
classes = ['dent', 'scratch', 'crack', 'glass_shatter', 'lamp_broken', 'tire_flat']
for cls in classes:
    (processed_dir / cls).mkdir(exist_ok=True)

# Process each split
total_copied = 0
for split in ['train2017', 'val2017', 'test2017']:
    csv_file = Path('/home/khalil_b/test_pfe/data/annotations_csv') / f'{split}_annotations.csv'
    if not csv_file.exists():
        print(f"Error: CSV file {csv_file} not found. Aborting {split} processing.", file=sys.stderr)
        continue
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            category_id = int(row['category_id'])
            if 1 <= category_id <= 6:  # Ensure valid category
                class_name = classes[category_id - 1]  # Map 1-6 to class names
                src_path = raw_dir / split / row['file_name']
                dest_path = processed_dir / class_name / row['file_name']
                print(f"Processing: {src_path} -> {dest_path}")
                if not src_path.exists():
                    print(f"Error: Source image {src_path} not found. Skipping.", file=sys.stderr)
                    continue
                try:
                    shutil.copy(src_path, dest_path)
                    print(f"Successfully copied {row['file_name']} to {class_name} ({split})")
                    total_copied += 1
                except Exception as e:
                    print(f"Error copying {src_path} to {dest_path}: {e}", file=sys.stderr)
            else:
                print(f"Warning: Invalid category_id {category_id} for {row['file_name']}. Skipping.", file=sys.stderr)

print(f"Image processing complete. Total images copied: {total_copied}. Check class distribution.")