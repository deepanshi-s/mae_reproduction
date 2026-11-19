import torch
import os
import io

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import time
import torchvision.transforms.functional as F
from torchvision.transforms import *
from torch.utils.data.distributed import DistributedSampler
from torchcodec.decoders import decode_png, decode_jpeg


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std =  np.array([0.229, 0.224, 0.225])

def detect_image_format(data: bytes) -> str:
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'PNG'
    elif data.startswith(b'\xff\xd8\xff'):
        return 'JPEG'
    return 'UNKNOWN'

class Imagenet100(Dataset):
    def __init__(self, dataset_dir: str, folder: str, img_transforms=None):
        super().__init__()
        files = os.listdir(os.path.join(dataset_dir, folder))
        all_dfs = []
        
        for f in files:
            file_path = os.path.join(dataset_dir, folder, f)
            
            df = pd.read_parquet(file_path)
            all_dfs.append(df)
        
        master_df = pd.concat(all_dfs, ignore_index=True)
        self.images = master_df['image'].tolist()
        self.labels = master_df['label'].tolist()
        
        self.total_data = len(master_df)
        self.img_transforms = img_transforms


    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        
        image_data = self.images[idx]
        label = self.labels[idx]
        
        raw_bytes = image_data['bytes'] if isinstance(image_data, dict) else image_data

        if isinstance(raw_bytes, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1)
        elif detect_image_format(raw_bytes) == 'PNG':
            img = decode_png(raw_bytes, mode='UNCHANGED', output_dtype=torch.uint8)
        elif detect_image_format(raw_bytes) == 'JPEG':
            img = decode_jpeg(raw_bytes, mode='UNCHANGED', output_dtype=torch.uint8)
        else:
            img = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
            img = torch.from_numpy(img).permute(2, 0, 1)

        if img.shape[0] != 3:
            img = img.repeat(3, 1, 1)

        img_tensor = self.img_transforms(img)

        return img_tensor, label

def init_transforms():
    train_transforms = v2.Compose([
        v2.RandomResizedCrop((224, 224), (0.2, 1.0)),
        v2.RandomHorizontalFlip(),
        
    ])
    val_transforms =  v2.Compose([
        v2.Resize((224, 224)),
        
    ])
    return train_transforms, val_transforms

def get_dataloaders(dataset_dir, train_folder, val_folder, batch_size, is_distributed):
    train_transforms, val_transforms = init_transforms()
    train_dataset = Imagenet100(dataset_dir, train_folder, train_transforms)
    val_dataset = Imagenet100(dataset_dir, val_folder, val_transforms)

    if is_distributed:
        train_distributed = DistributedSampler(train_dataset)
        val_distrbuted = DistributedSampler(val_dataset)
        shuffle = False
    else:
        train_distributed = None
        val_distrbuted = None
        shuffle = True

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, prefetch_factor=2, num_workers=4, sampler=train_distributed, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, prefetch_factor=2, num_workers=4, sampler=val_distrbuted, pin_memory=True)

    return train_dataloader, val_dataloader