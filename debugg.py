import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil

# Load metadata
metadata = pd.read_csv('HAM10000_metadata.csv')

# Create directories for train and test sets
os.makedirs('data/train/images', exist_ok=True)
os.makedirs('data/train/labels', exist_ok=True)
os.makedirs('data/val/images', exist_ok=True)
os.makedirs('data/val/labels', exist_ok=True)

# Split the dataset into train and validation sets
train_df, val_df = train_test_split(metadata, test_size=0.2, stratify=metadata['dx'])

# Function to copy images and masks to respective directories
def copy_files(df, subset):
    for index, row in df.iterrows():
        image_file = f'ham10000_combined/{row["image_id"]}.jpg'
        mask_file = f'ham10000_segmentations/{row["image_id"]}.png'
        if os.path.exists(image_file) and os.path.exists(mask_file):
            shutil.copy(image_file, f'data/{subset}/images/{row["image_id"]}.jpg')
            shutil.copy(mask_file, f'data/{subset}/labels/{row["image_id"]}.png')

# Copy files to train and validation directories
copy_files(train_df, 'train')
copy_files(val_df, 'val')


import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

# Define augmentations
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.OneOf([
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.PiecewiseAffine(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
    ToTensorV2()
])

# Apply augmentations and save augmented images
def augment_and_save(image_path, mask_path, save_dir, transform):
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))
    augmented = transform(image=image, mask=mask)
    aug_image = Image.fromarray(augmented['image'])
    aug_mask = Image.fromarray(augmented['mask'])
    aug_image.save(os.path.join(save_dir, 'images', os.path.basename(image_path)))
    aug_mask.save(os.path.join(save_dir, 'labels', os.path.basename(mask_path)))

# Apply augmentations to train set
for index, row in train_df.iterrows():
    image_path = f'data/train/images/{row["image_id"]}.jpg'
    mask_path = f'data/train/labels/{row["image_id"]}.png'
    augment_and_save(image_path, mask_path, 'data/train', transform)


import cv2

def create_yolo_annotation(image_id, mask_path, save_dir):
    mask = cv2.imread(mask_path, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    with open(os.path.join(save_dir, 'labels', f'{image_id}.txt'), 'w') as f:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_center = (x + w / 2) / mask.shape[1]
            y_center = (y + h / 2) / mask.shape[0]
            width = w / mask.shape[1]
            height = h / mask.shape[0]
            f.write(f'0 {x_center} {y_center} {width} {height}\n')

# Create annotations for train and validation sets
for index, row in train_df.iterrows():
    mask_path = f'data/train/labels/{row["image_id"]}.png'
    create_yolo_annotation(row["image_id"], mask_path, 'data/train')

for index, row in val_df.iterrows():
    mask_path = f'data/val/labels/{row["image_id"]}.png'
    create_yolo_annotation(row["image_id"], mask_path, 'data/val')


!git clone https://github.com/WongKinYiu/yolov7
%cd yolov7
!pip install -r requirements.txt

# Create a custom YAML file for the dataset
dataset_yaml = """
train: ../data/train/images
val: ../data/val/images

nc: 1
names: ['skin_disease']
"""

with open('data/skin_disease.yaml', 'w') as f:
    f.write(dataset_yaml)

# Train the YOLOv7 model
!python train.py --img 640 --batch 16 --epochs 50 --data data/skin_disease.yaml --cfg cfg/training/yolov7.yaml --weights 'yolov7.pt' --name yolov7_skin_disease


