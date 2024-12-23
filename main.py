import torch
import torch.nn as nn
from torchvision import models


from Utils import (
    logger
)

from Utils.config import (
    DO_ATTACK,
    DO_DEFENSE,
    IMAGE_PATH,
    MODELNAME,
    LOG_PATH,
)



if __name__ == "__main__":

    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    logger.info("-" * 50)
    logger.info("Executing main")

    # Load pretrained Model
    model = getattr(models, MODELNAME)(pretrained=True)
    logger.info("-" * 50)
    logger.info("Model loaded")


