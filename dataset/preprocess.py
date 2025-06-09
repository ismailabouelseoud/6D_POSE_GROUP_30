# FILE: dataset/preprocess.py
import os
import yaml
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
from ruamel.yaml import YAML
import sys

# NOTE: This is a standalone script to be run once.
# It requires the original Linemod_preprocessed.zip file.

# --- Configuration ---
LINEMOD_RAW_ROOT = "/content/Linemod_preprocessed" # Directory where zip is extracted
YOLO_DATASET_ROOT = "/content/datasets/linemod/Linemod_preprocessed_yolo_2" # Target directory

OBJECT_IDS = ['01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '14', '15']
OBJECT_NAMES = ['ape', 'benchvise', 'camera', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
TRAIN_RATIO = 0.85 # Using more for training as per many repos
RANDOM_SEED = 42
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

def create_yolo_directories():
    """Creates the necessary directory structure for the YOLO dataset."""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(YOLO_DATASET_ROOT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_ROOT, 'labels', split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_ROOT, 'depth', split), exist_ok=True)

def convert_bbox_to_yolo(bbox):
    """Converts [x_min, y_min, w, h] to normalized [center_x, center_y, w, h]."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / IMAGE_WIDTH
    cy = (y + h / 2.0) / IMAGE_HEIGHT
    wn = w / IMAGE_WIDTH
    hn = h / IMAGE_HEIGHT
    return cx, cy, wn, hn

def process_and_split_data():
    """Main function to process all data, create labels, and split the dataset."""
    all_files_map = {}
    object_id_to_class_id = {obj_id: i for i, obj_id in enumerate(OBJECT_IDS)}

    print("Gathering all image paths...")
    for obj_id in OBJECT_IDS:
        rgb_folder = os.path.join(LINEMOD_RAW_ROOT, 'data', obj_id, 'rgb')
        all_files_map[obj_id] = sorted(os.listdir(rgb_folder))

    print("Splitting data into train/val/test sets...")
    train_files, val_files, test_files = {}, {}, {}
    for obj_id, files in all_files_map.items():
        train_val, test = train_test_split(files, test_size=0.15, random_state=RANDOM_SEED)
        train, val = train_test_split(train_val, test_size=0.1, random_state=RANDOM_SEED)
        train_files[obj_id] = train
        val_files[obj_id] = val
        test_files[obj_id] = test

    splits = {'train': train_files, 'val': val_files, 'test': test_files}
    total_processed = 0

    for split_name, split_data in splits.items():
        print(f"\nProcessing '{split_name}' split...")
        for obj_id, files in split_data.items():
            gt_path = os.path.join(LINEMOD_RAW_ROOT, 'data', obj_id, 'gt.yml')
            gt_data = yaml.safe_load(open(gt_path))
            class_id = object_id_to_class_id[obj_id]

            for filename in files:
                sample_id = int(os.path.splitext(filename)[0])
                if sample_id not in gt_data: continue

                annotations = gt_data[sample_id]
                for ann in annotations:
                    if ann['obj_id'] != int(obj_id): continue

                    # Create label file content
                    bbox_yolo = convert_bbox_to_yolo(ann['obj_bb'])
                    cam_r = ' '.join(map(str, ann['cam_R_m2c']))
                    cam_t = ' '.join(map(str, ann['cam_t_m2c']))
                    
                    label_content = f"{class_id+1} {' '.join(map(str, bbox_yolo))}\n"
                    label_content += f"{class_id+1} {cam_r}\n"
                    label_content += f"{class_id+1} {cam_t}\n"
                    
                    # Define paths
                    new_filename_base = f"{obj_id}_{filename.split('.')[0]}"
                    img_src = os.path.join(LINEMOD_RAW_ROOT, 'data', obj_id, 'rgb', filename)
                    depth_src = os.path.join(LINEMOD_RAW_ROOT, 'data', obj_id, 'depth', filename)
                    
                    img_dst = os.path.join(YOLO_DATASET_ROOT, 'images', split_name, f"{new_filename_base}.png")
                    depth_dst = os.path.join(YOLO_DATASET_ROOT, 'depth', split_name, f"{new_filename_base}.png")
                    label_dst = os.path.join(YOLO_DATASET_ROOT, 'labels', split_name, f"{new_filename_base}.txt")

                    # Copy/link files and write label
                    shutil.copy(img_src, img_dst)
                    if os.path.exists(depth_src): shutil.copy(depth_src, depth_dst)
                    with open(label_dst, 'w') as f:
                        f.write(label_content)

                    total_processed += 1
    print(f"\nSuccessfully processed {total_processed} annotations.")

def create_data_yaml():
    """Creates the data.yaml file for YOLO training."""
    yolo_config = {
        'train': os.path.abspath(os.path.join(YOLO_DATASET_ROOT, 'images/train')),
        'val': os.path.abspath(os.path.join(YOLO_DATASET_ROOT, 'images/val')),
        'test': os.path.abspath(os.path.join(YOLO_DATASET_ROOT, 'images/test')),
        'nc': len(OBJECT_IDS),
        'names': OBJECT_NAMES
    }
    with open(os.path.join(YOLO_DATASET_ROOT, 'data.yaml'), 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
    print("✓ data.yaml created.")

def main():
    """Main preprocessing pipeline."""
    print("Starting Linemod to YOLO format conversion...")
    if not os.path.exists(LINEMOD_RAW_ROOT):
        print(f"Error: Raw Linemod directory not found at {LINEMOD_RAW_ROOT}")
        print("Please extract Linemod_preprocessed.zip to that location.")
        return

    create_yolo_directories()
    process_and_split_data()
    create_data_yaml()
    # Manual step: copy models folder from raw to YOLO root
    shutil.copytree(os.path.join(LINEMOD_RAW_ROOT, 'models'), os.path.join(YOLO_DATASET_ROOT, 'pose_models'))
    print("✓ Preprocessing complete. Models folder also copied.")

if __name__ == '__main__':
    # This script should be run from the root of the repository.
    # Ensure the LINEMOD_RAW_ROOT path points to your extracted data.
    main()
