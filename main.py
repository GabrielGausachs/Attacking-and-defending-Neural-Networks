import torch
import torch.nn as nn
from torchvision import models


from Utils import (
    logger,
    dataloader,
    attacking
)

from Utils.config import (
    DO_ATTACK,
    DO_DEFENSE,
    IMAGE_PATH,
    MODELNAME,
    LOG_PATH,
    ATTACK_NAME,
    IMAGES_TO_TEST,
    CRITERION,
    EPSILON,
    STEPSIZE,
    NUM_ITERATIONS,
    DEVICE
)

# Criterion
criterion = {
    "MSELoss": torch.nn.MSELoss(),
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss()
    }



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
    testloader = dataloader.CustomDataloader().create_dataloaders()
    logger.info("-" * 50)
    logger.info("Data loaded")

    if DO_ATTACK == True:
        # Attacking
        logger.info("-" * 50)
        logger.info("Start attacking")

        # Create a criterion object
        criterion = criterion[CRITERION]

        for i, (image, label) in enumerate(testloader):

            if ATTACK_NAME == 'ifgsm':
                logger.info("Attacking using I-FGSM")
                logger.info(f"epsilon: {EPSILON}, stepsize: {STEPSIZE}, num_iter: {NUM_ITERATIONS}")
                # Attacking with I-FGSM
                adversial_image = attacking.I_FGM_attack(model,image,label,criterion,EPSILON,STEPSIZE,NUM_ITERATIONS)

            if i >= IMAGES_TO_TEST:
                break
        




        



