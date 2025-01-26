from torch.utils.data import Dataset
from Utils.logger import initialize_logger, get_logger
import numpy as np
import torch
import os
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms

logger = get_logger()


class CustomDataset(Dataset):
    def __init__(self,images_path,labels_path,transform=None):

        self.images_dir = images_path
        self.labels_file = labels_path
        self.transform = transform

        self.image_files = sorted(os.listdir(images_path))

        # Load labels from the .txt file
        with open(self.labels_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f]

        if len(self.image_files) != len(self.labels):
            raise ValueError("Number of images and labels do not match!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image
        image_path = os.path.join(self.images_dir, self.image_files[index])
        image = Image.open(image_path)

        # Convert images that are grayscale to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Load label
        label = self.labels[index]
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Transform the image if it is necessary
        if self.transform:
            image = self.transform(image)

        return image,label_tensor
