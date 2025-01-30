import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from Utils.logger import initialize_logger, get_logger
from Utils.dataset import CustomDataset, CustomAdvDataset
from torchvision import transforms

from Utils.config import (
    IMAGE_PATH,
    BATCH_SIZE_ATTACK,
    NUM_WORKERS,
    LABELS_PATH,
    NUM_LABELS_PATH,
    BATCH_SIZE_UNET,
    ADV_PATH,
    RANDOM_SEED
)


logger = get_logger()


class CustomDataloader:
    def __init__(self):

        self.batch_size_attack = BATCH_SIZE_ATTACK
        self.batch_unet = BATCH_SIZE_UNET
        self.num_workers = NUM_WORKERS
        self.images_path = IMAGE_PATH
        self.num_labels_path = NUM_LABELS_PATH
        self.labels_path = LABELS_PATH
        self.images_adv = ADV_PATH


    def create_dataloaders(self):

        logger.info("-" * 50)
        logger.info(f'Reading the data from {self.images_path}...')



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

        dataset = CustomDataset(self.images_path,self.num_labels_path,self.labels_path,transform)

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

        
        torch.manual_seed(RANDOM_SEED)

        train_size = int(0.8 * len(dataset))
        test_size = int(0.05 * len(dataset))
        val_size = len(dataset) - train_size - test_size

        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        logger.info(f"Dataset split: {train_size} samples for training, {val_size} samples for validation, {test_size} samples for testing.")

        # Create DataLoaders for train, validation, and test datasets
        logger.info("-" * 50)
        logger.info('Creating dataloaders...')

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_UNET, shuffle=True, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE_UNET, shuffle=False, num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE_UNET, shuffle=False, num_workers=NUM_WORKERS)

        # Gather info about DataLoaders
        dataloader_info = {
            'Train loader': {
                'Number of samples': len(train_dataloader.dataset),
                'Batch size': train_dataloader.batch_size,
                'Number of batches': len(train_dataloader),
            },
            'Validation loader': {
                'Number of samples': len(val_dataloader.dataset),
                'Batch size': val_dataloader.batch_size,
                'Number of batches': len(val_dataloader),
            },
            'Test loader': {
                'Number of samples': len(test_dataloader.dataset),
                'Batch size': test_dataloader.batch_size,
                'Number of batches': len(test_dataloader),
            }
        }

        logger.info(f"Loader info: {dataloader_info}")

        return train_dataloader, val_dataloader, test_dataloader








