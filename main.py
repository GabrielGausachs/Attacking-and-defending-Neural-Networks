import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import os
import csv
from PIL import Image 
import pandas as pd


from Utils import (
    logger,
    dataloader,
    attacking,
    visualize,
    utils,
    train,
)

from Models import (
    UNet
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
    DEVICE,
    ADV_PATH,
    OPTIMIZER,
    LEARNING_RATE,
    EPOCHS,
)

# Criterion
criterion = {
    "MSELoss": torch.nn.MSELoss(),
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss()
    }

# Optimizers
optimizers = {
    "Adam": torch.optim.Adam
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

    # Create folder to save adversarial images
    os.makedirs(ADV_PATH, exist_ok=True)
    logger.info(f"Adversarial images will be saved in: {ADV_PATH}")

    # Create metadata file to save the labels
    metadata_file = os.path.join(ADV_PATH, "metadata.xlsx")

    data = {
    "Image_Name": [],
    "True_Label": [],
    "Predicted_Label": []
    }

    # Create a DataFrame
    df = pd.DataFrame(data)


    if DO_ATTACK == True:
        # Attacking
        logger.info("-" * 50)
        logger.info("Start attacking")
        logger.info("-" * 50)

        # Create a criterion object
        criterion = criterion[CRITERION]

        image_count = 0

        for i, (image, label) in enumerate(val_loader):

            # Log image size
            logger.info(f"Image size: {image.size()}")  

            if ATTACK_NAME == 'ifgsm':
                logger.info("-" * 50)
                logger.info("Attacking using I-FGSM")
                logger.info(f"epsilon: {EPSILON}, stepsize: {STEPSIZE}, num_iter: {NUM_ITERATIONS}")
                logger.info("-" * 50)
                # Attacking with I-FGSM
                adversial_image, pred_label = attacking.I_FGM_attack(model,image,label,criterion,EPSILON,STEPSIZE,NUM_ITERATIONS)
                #visualize.plot_images(image,adversial_image,label.item(),pred_label)
                image_count,df = utils.save_adversial_images(adversial_image,label,pred_label,ADV_PATH,df,image_count)

            if image_count >= IMAGES_TO_TEST:
                logger.info("Reached 15,000 images. Stopping attack.")
                df.to_excel(metadata_file, index=False)
                logger.info("Metadata saved in Excel File")
                break

    if DO_DEFENSE == True:
        # Attacking
        logger.info("-" * 50)
        logger.info("Start defending")
        logger.info("-" * 50)

        # Reading adversial images
        train_adv_loader, val_adv_loader = dataloader.CustomDataloader().dataloader_adv()
        logger.info("-" * 50)
        logger.info("Adversial data loaded")

        # Training DUNet
        Dunet = UNet.DUNET(3,3)

        # Create an optimizer object
        optimizer = optimizers[OPTIMIZER](model.parameters(), lr=LEARNING_RATE)

        # Create a criterion object
        criterion = criterion[CRITERION]

        logger.info("-" * 50)
        num_params = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Starting training with DUNET that has {num_params} parameters")
        logger.info(f"Learning rate: {LEARNING_RATE}")

        for epoch in range(EPOCHS):
            logger.info(f"--- Epoch: {epoch} ---")
            epoch_loss_train, epoch_metric_train = train.train(
                model=Dunet,
                loader=train_adv_loader,
                target_model = model,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
                epochs=EPOCHS
            )





        




        



