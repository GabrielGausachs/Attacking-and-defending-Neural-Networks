import tarfile
import os
import torch
import tarfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Utils.logger import initialize_logger, get_logger

from Utils.config import (
    IMAGE_PATH,
    BATCH_SIZE_ATTACK,
    NUM_WORKERS
)



logger = get_logger()


class CustomDataloader:
    def __init__(self):

        self.batch_size_attack = BATCH_SIZE_ATTACK
        self.num_workers = NUM_WORKERS
        self.images_path = IMAGE_PATH


    def create_dataloaders(self):

        logger.info("-" * 50)
        logger.info(f'Reading the data from {self.images_path}...')


        # Extract files from tar folder - Only if you have the validation folder as .tar. DO IT ONLY ONCE!
        #self.reading_tarfiles()

        # Create dataset
        logger.info("-" * 50)
        logger.info('Creating dataset...')

        dataset = datasets.ImageFolder(root=self.images_path)

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








