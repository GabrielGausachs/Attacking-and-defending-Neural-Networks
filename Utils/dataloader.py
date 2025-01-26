import tarfile
import os
import torch
import tarfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Utils.logger import initialize_logger, get_logger
from Utils.dataset import CustomDataset, CustomAdvDataset
from torchvision import transforms

from Utils.config import (
    IMAGE_PATH,
    BATCH_SIZE_ATTACK,
    NUM_WORKERS,
    LABEL_PATH,
    BATCH_SIZE_UNET,
    ADV_PATH,
)


logger = get_logger()


class CustomDataloader:
    def __init__(self):

        self.batch_size_attack = BATCH_SIZE_ATTACK
        self.batch_unet = BATCH_SIZE_UNET
        self.num_workers = NUM_WORKERS
        self.images_path = IMAGE_PATH
        self.labels_path = LABEL_PATH
        self.images_adv = ADV_PATH


    def create_dataloaders(self):

        logger.info("-" * 50)
        logger.info(f'Reading the data from {self.images_path}...')


        # Extract files from tar folder - Only if you have the validation folder as .tar. DO IT ONLY ONCE!
        #self.reading_tarfiles()

        # Create dataset
        logger.info("-" * 50)
        logger.info('Creating dataset...')

        transform = transforms.Compose([
            transforms.Resize(256),                       # Resize the shorter side to 256
            transforms.CenterCrop(224),                   # Center crop to 224x224
            transforms.ToTensor(),                        # Convert image to tensor (from [0, 255] to [0, 1])
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet's mean and std
                                std=[0.229, 0.224, 0.225]),
        ])

        dataset = CustomDataset(self.images_path,self.labels_path,transform)
        print("Number of classes:", len(dataset.classes))

        # Create DataLoader
        logger.info("-" * 50)
        logger.info('Creating dataloader...')
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_ATTACK, shuffle=True, num_workers=NUM_WORKERS)

        # Gather info about Dataloader
        dataloader_info = {
            'Number of samples': len(dataloader.dataset),
            'Batch size': dataloader.batch_size,
            'Number of batches': len(dataloader)
        }

        logger.info(f"Val loader info: {dataloader_info}")

        return dataloader
    

    def reading_tarfiles(self):
        logger.info("Extracting files from .tar")
        self.images_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"ILSVRC2012_img_val.tar")
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"ILSVRC2012_img_val"), exist_ok=True)
        with tarfile.open(self.images_path, 'r:*') as tar:
            tar.extractall(path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"ILSVRC2012_img_val"))
            logger.info(f"Extracted {self.images_path} to {os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"ILSVRC2012_img_val")}")
        self.images_path = IMAGE_PATH
    
    def dataloader_adv(self):

        logger.info("-" * 50)
        logger.info(f'Reading the data from {self.images_adv}...')

        # Create dataset
        logger.info("-" * 50)
        logger.info('Creating adv dataset...')

        transform = transforms.Compose([
            transforms.Resize(256),                       
            transforms.CenterCrop(224),                  
            transforms.ToTensor(),                        
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        dataset = CustomAdvDataset(self.images_adv,transform)

        # Create DataLoader
        logger.info("-" * 50)
        logger.info('Creating dataloader...')
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_UNET, shuffle=True, num_workers=NUM_WORKERS)

        # Gather info about Dataloader
        dataloader_info = {
            'Number of samples': len(dataloader.dataset),
            'Batch size': dataloader.batch_size,
            'Number of batches': len(dataloader)
        }

        logger.info(f"Val loader info: {dataloader_info}")

        return dataloader








