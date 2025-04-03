import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import numpy as np
from torchvision import datasets
import os
import tempfile
import boto3
from io import BytesIO
from PIL import Image

# AWS S3 Configuration
S3_BUCKET = "sea-guardian-image"  # Your bucket name
S3_FOLDER = "dataset/"  # Folder in your bucket

# Transformation Pipeline
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15),
    A.RandomBrightnessContrast(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=None, min_height=8, min_width=8, fill_value=0, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def download_s3_file(s3_client, s3_key, local_path):
    """Downloads a file from S3 to a local path."""
    try:
        s3_client.download_file(S3_BUCKET, s3_key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading {s3_key}: {e}")
        return False

def load_data(batch_size=32):
    local_temp_dir = tempfile.mkdtemp()
    s3_client = boto3.client("s3")

    # Download all files from S3
    for root, dirs, files in os.walk("dataset"): # using dataset as a source to get the names of the files.
        for file in files:
            s3_key = os.path.join(S3_FOLDER, os.path.join(root, file).replace('dataset\\', ''))
            local_file_path = os.path.join(local_temp_dir, os.path.join(root, file).replace('dataset\\', ''))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            download_s3_file(s3_client, s3_key, local_file_path)

    dataset = datasets.ImageFolder(root=local_temp_dir, transform=lambda x: transform(image=np.array(x))["image"])

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader