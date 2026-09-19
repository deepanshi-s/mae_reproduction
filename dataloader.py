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

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std =  np.array([0.229, 0.224, 0.225])

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
            img = np.array(Image.fromarray(raw_bytes.astype('uint8')).convert("RGB"))
        else:
            img = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
        
        img = img/255.
        img_tensor = self.img_transforms(img)

        img_tensor = img_tensor.to(torch.float)
        return img_tensor, label

def init_transforms():
    train_transforms = Compose([
        ToTensor(),
        RandomResizedCrop((224, 224), (0.2, 1.0)),
        RandomHorizontalFlip(),
        Normalize(imagenet_mean, imagenet_std),
    ])
    val_transforms =  Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize(imagenet_mean, imagenet_std),
    ])
    return train_transforms, val_transforms

def get_dataloaders(dataset_dir, train_folder, val_folder, batch_size):
    train_transforms, val_transforms = init_transforms()
    train_dataset = Imagenet100(dataset_dir, train_folder, train_transforms)
    val_dataset = Imagenet100(dataset_dir, val_folder, val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader