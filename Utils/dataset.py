from torch.utils.data import Dataset
from Utils.logger import initialize_logger, get_logger
import numpy as np
import torch
import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import pandas as pd

logger = get_logger()


class CustomDataset(Dataset):
    def __init__(self,images_path,num_labels_path,labels_path,transform=None):

        self.images_dir = images_path
        self.num_labels_path = num_labels_path
        self.labels_file = labels_path
        self.transform = transform

        self.image_files = os.listdir(images_path)

        # CHANGE THIS!!
        # Load the JSON mapping file
        with open(labels_path, "r") as f:
            self.label_map = json.load(f)

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
        xml_file = os.path.join(self.num_labels_path, os.path.splitext(self.image_files[index])[0] + ".xml")
        class_id = self._parse_xml(xml_file)
        if class_id is None:
            raise ValueError(f"Could not find class ID for image: {self.image_files[index]}")
        
        real_name = self._get_real_name(class_id)
        if real_name is None:
            raise ValueError(f"Class ID {class_id} not found in label map")

        # Transform the image if it is necessary
        if self.transform:
            image = self.transform(image)

        return image, real_name
    
    def _parse_xml(self, xml_path):
        """Parse the XML file to extract the class label."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Extract the <name> tag inside <object>
        object_tag = root.find("object")
        if object_tag is not None:
            name_tag = object_tag.find("name")
            if name_tag is not None:
                return name_tag.text  # E.g., "n01751748"
        return None
    
    def _get_real_name(self, class_id):
        """Map the class ID to the real name using the label map."""
        for key, value in self.label_map.items():
            if value[0] == class_id:
                return int(key)  # Return the real name (e.g., 359)
        return None

class CustomAdvDataset(Dataset):
    def __init__(self, adv_path, transform=None):

        self.adv_path = adv_path
        self.labels_excel_path = os.path.join(adv_path, "metadata.xlsx")
        self.transform = transform

        # Load labels from the Excel file
        self.labels_df = pd.read_excel(self.labels_excel_path)


        # Get the labels
        self.image_labels = []
        self.predicted_labels = []
        for _,row in self.labels_df.iterrows():
            self.image_labels.append(int(row['True_Label']))
            self.predicted_labels.append(int(row['Predicted_Label']))
            

        # Get list of image files
        self.image_files = [f for f in os.listdir(adv_path) if os.path.isfile(os.path.join(adv_path, f)) and not f.lower().endswith('.xlsx')]


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

        # Load predicted label
        pred_label = self.predicted_labels[index]
        pred_label_tensor = torch.tensor(pred_label, dtype=torch.long)

        # Transform the image if it is necessary
        if self.transform:
            image = self.transform(image)

        return image, label_tensor, pred_label_tensor
