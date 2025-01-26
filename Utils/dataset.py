from torch.utils.data import Dataset
from Utils.logger import initialize_logger, get_logger
import numpy as np
import torch
import os
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import pandas as pd

logger = get_logger()


class CustomDataset(Dataset):
    def __init__(self,images_path,labels_path,transform=None):

        self.images_dir = images_path
        self.labels_file = labels_path
        self.transform = transform

        self.image_files = sorted(os.listdir(images_path))

        # Load labels from the .txt file
        with open(self.labels_file, 'r') as f:
            self.labels = [int(line.strip()) - 1 for line in f] # making labels 0-indexed

        self.classes = sorted(list(set(self.labels)))

        if 0 in self.classes:
            print("Label 0 exists in self.classes.")
        else:
            print("Label 0 does NOT exist in self.classes.")

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

class CustomAdvDataset(Dataset):
    def __init__(self, adv_path, transform=None):

        self.adv_path = adv_path
        self.labels_csv_path = os.path.join(adv_path, "metadata.csv")
        self.transform = transform

        # Load labels from the CSV file
        self.labels_df = pd.read_csv(self.labels_csv_path)

        # Get the labels
        self.image_labels = []
        for _,row in self.labels_df.iterrows():
            self.image_labels.append(int(row['True_Label']))
            

        # Get list of image files
        self.image_files = [f for f in os.listdir(adv_path) if os.path.isfile(os.path.join(adv_path, f))]

        # Ensure number of images matches number of labels
        if len(self.image_files) != len(self.image_labels):
            raise ValueError("Number of images and labels do not match!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image
        image_name = self.image_files[index]
        image_path = os.path.join(self.adv_path, image_name)
        image = Image.open(image_path)

        # Convert images that are grayscale to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Load label
        label = self.image_labels[index]
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Transform the image if it is necessary
        if self.transform:
            image = self.transform(image)

        return image, label_tensor
