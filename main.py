import torch
import torch.nn as nn
from torchvision import models


from Utils import (
    logger,
    dataloader
)

from Utils.config import (
    DO_ATTACK,
    DO_DEFENSE,
    IMAGE_PATH,
    MODELNAME,
    LOG_PATH,
    DEVICE
)



if __name__ == "__main__":

    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    logger.info("-" * 50)
    logger.info("Executing main")

    # Load pretrained Model
    model = getattr(models, MODELNAME)(pretrained=True)
    model.to(DEVICE)
    logger.info("-" * 50)
    logger.info("Model loaded")

    # Load the data
    train_loader, val_loader = dataloader.CustomDataloader().create_dataloaders()
    logger.info("-" * 50)
    logger.info("Data loaded")

    if DO_ATTACK == True:
        # Attacking
        logger.info("-" * 50)
        logger.info("Start attacking")
        



