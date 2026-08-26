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

class Imagenet100(Dataset):
    def __init__(self, dataset_dir: str, folder: str, img_transforms=None):
        super().__init__()
        files = os.listdir(os.path.join(dataset_dir, folder))
        all_dfs = []
        
        # 1. Load everything into RAM immediately
        for f in files:
            file_path = os.path.join(dataset_dir, folder, f)
            
            df = pd.read_parquet(file_path)
            all_dfs.append(df)
        
        master_df = pd.concat(all_dfs, ignore_index=True)
        self.images = master_df['image'].tolist()
        self.labels = master_df['label'].tolist()
        
        self.total_data = len(master_df)


    def __len__(self):
        return self.total_data

    def __getitem__(self, idx):
        
        image_data = self.images[idx]
        label = self.labels[idx]
        
        raw_bytes = image_data['bytes'] if isinstance(image_data, dict) else image_data
        
        if isinstance(raw_bytes, np.ndarray):
            img = np.array(Image.fromarray(raw_bytes.astype('uint8')))
        else:
            img = np.array(Image.open(io.BytesIO(raw_bytes)))
        
        img = np.resize(img, (224, 224, 3))
        return img, label


def get_dataloaders(dataset_dir, train_folder, val_folder, batch_size):
    train_dataset = Imagenet100(dataset_dir, train_folder)
    val_dataset = Imagenet100(dataset_dir, val_folder)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader