import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import os
import torchvision.transforms as T
from PIL import Image

class AptosDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0] + '.png'
        img_path = os.path.join(self.root, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx, 1]
        return image, label