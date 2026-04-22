import torch

from torch.utils.data import Dataset
import cv2
import numpy as np

class MyDataset(Dataset):
    def __init__(self, image_paths, masks_paths):
        self.image_paths = image_paths
        self.masks_paths = masks_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_LINEAR)
        img = img.astype("float32") / 255.0  # Normalization is key for mIOU
        
        mask = cv2.imread(self.masks_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)
        
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        
        return img, mask