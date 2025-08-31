import json
import csv
from pathlib import Path
from pycocotools.coco import COCO

# Paths
raw_dir = Path('../data/raw/CarDD_release/CarDD_COCO')
output_dir = Path('../data/annotations_csv')
output_dir.mkdir(exist_ok=True)

# Process each split
for split in ['train2017', 'val2017', 'test2017']:
    coco = COCO(str(raw_dir / 'annotations' / f'instances_{split}.json'))
    
    # Prepare CSV data
    csv_data = []
    for img_id in coco.imgs:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        if anns:
            # Use annotation with largest area as primary label
            primary_ann = max(anns, key=lambda ann: ann['area'])
            csv_data.append({
                'image_id': img_info['id'],
                'file_name': img_info['file_name'],
                'category_id': primary_ann['category_id'],
                'area': primary_ann['area'],
                'bbox': str(primary_ann['bbox'])  # Convert list to string for CSV
            })
        else:
            # Handle images with no annotations (optional, skip for now)
            continue

    # Write to CSV
    csv_file = output_dir / f'{split}_annotations.csv'
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['image_id', 'file_name', 'category_id', 'area', 'bbox']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Converted {split} annotations to {csv_file}")