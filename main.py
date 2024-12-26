import torch
import torch.nn as nn
from torchvision import models


from Utils import (
    logger,
    dataloader,
    attacking,
    visualize,
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
    val_loader = dataloader.CustomDataloader().create_dataloaders()
    logger.info("-" * 50)
    logger.info("Data loaded")

    if DO_ATTACK == True:
        # Attacking
        logger.info("-" * 50)
        logger.info("Start attacking")

        # Create a criterion object
        criterion = criterion[CRITERION]

        for i, (image, label) in enumerate(val_loader):

            logger.info(f"Label: {label}")

            # Log image size
            logger.info(f"Image size: {image.size()}")  

            if ATTACK_NAME == 'ifgsm':
                logger.info("Attacking using I-FGSM")
                logger.info(f"epsilon: {EPSILON}, stepsize: {STEPSIZE}, num_iter: {NUM_ITERATIONS}")
                # Attacking with I-FGSM
                adversial_image, pred_label = attacking.I_FGM_attack(model,image,label,criterion,EPSILON,STEPSIZE,NUM_ITERATIONS)
                visualize.plot_images(image,adversial_image,label,pred_label)

            if i >= IMAGES_TO_TEST:
                break
        




        



