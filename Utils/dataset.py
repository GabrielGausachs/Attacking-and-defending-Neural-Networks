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
        self.labels_dir = labels_path
        self.transform = transform

        self.image_files = sorted(os.listdir(images_path))
        self.label_files = sorted(os.listdir(labels_path))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image
        image_path = os.path.join(self.images_dir, self.image_files[index])
        image = Image.open(image_path)

        # Load label
        label_path = os.path.join(self.labels_dir, self.label_files[index])
        label = self.get_label_from_xml(label_path)

        # Transform the image if it is necessary
        if self.transform:
            image = self.transform(image)

        return image,label

    def get_label_from_xml(self,xml_file):
        #Parses an XML file to extract the label(s).

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract labels from objects
        label = next((obj.find("name").text for obj in root.findall("object")), None)

        return label
