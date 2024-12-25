from torch.utils.data import Dataset
from Utils.logger import initialize_logger, get_logger
import numpy as np
import torch
import os
from PIL import Image
import xml.etree.ElementTree as ET

logger = get_logger()


class CustomDataset(Dataset):
    def __init__(self,images_path,labels_path):

        self.images_dir = images_path
        self.labels_dir = labels_path

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
        label = self.parse_xml_label(label_path)

        return image,label

    def get_label_from_xml(xml_file):
        #Parses an XML file to extract the label(s).

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract labels from objects
        label = [obj.find("name").text for obj in root.findall("object")]

        return label
