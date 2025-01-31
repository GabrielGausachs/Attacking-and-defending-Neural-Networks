import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import os
from PIL import Image 
import pandas as pd
import json
from datetime import datetime


from Utils import (
    logger,
    dataloader,
    attacking,
    utils,
    train,
    val,
    savemodel
)

from Models import (
    UNet,
)

from Utils.config import (
    DO_ATTACK,
    DO_DEFENSE,
    MODELNAME,
    RESULTS_PATH,
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
    DEFENSE_MODEL,
    DO_TRAIN,
    DO_TEST,
    MODEL_SAVED_NAME,
    MODELS_PATH,
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

# Models
our_models = {"DUNet": UNet.DUNET}

if __name__ == "__main__":

    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    logger.info("-" * 50)
    logger.info("Executing main")

    # Load pretrained Model
    resnet = getattr(models, MODELNAME)(pretrained=True)
    resnet.to(DEVICE)
    logger.info("-" * 50)
    logger.info("Model loaded")


    if DO_ATTACK == True:

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
        "Predicted_Label_Previous": [],
        "Predicted_Label": []
        }

        # Create a DataFrame
        df = pd.DataFrame(data)
        # Attacking
        logger.info("-" * 50)
        logger.info("Start attacking")
        logger.info("-" * 50)

        # Create a criterion object
        criterion = criterion[CRITERION]

        image_count = 0
        new_rows = []

        for i, (image, label) in enumerate(val_loader):

            # Log image size
            logger.info(f"Image size: {image.size()}")  

            if ATTACK_NAME == 'ifgsm':
                logger.info("-" * 50)
                logger.info("Attacking using I-FGSM")
                logger.info(f"epsilon: {EPSILON}, stepsize: {STEPSIZE}, num_iter: {NUM_ITERATIONS}")
                logger.info("-" * 50)
                # Attacking with I-FGSM
                adversial_image, pred_label, pred_label_previous = attacking.I_FGM_attack(resnet,image,label,criterion,EPSILON,STEPSIZE,NUM_ITERATIONS)
                image_count,new_rows = utils.save_adversial_images(adversial_image,label,pred_label,pred_label_previous,ADV_PATH,new_rows,image_count)

            if image_count >= IMAGES_TO_TEST:
                logger.info("Reached 15,000 images. Stopping attack.")
                new_rows_df = pd.DataFrame(new_rows)
                # Concatenate the new rows with the existing DataFrame
                df = pd.concat([df, new_rows_df], ignore_index=True)
                df.to_excel(metadata_file, index=False)
                logger.info("Metadata saved in Excel File")
                break

    if DO_DEFENSE == True:
        # Defending
        logger.info("-" * 50)
        logger.info("Start defending")
        logger.info("-" * 50)

        # Reading adversial images
        train_adv_loader, val_adv_loader, test_adv_loader = dataloader.CustomDataloader().dataloader_adv()
        logger.info("-" * 50)
        logger.info("Adversial data loaded")

        # DUNet
        Dunet = our_models[DEFENSE_MODEL](3, 3).to(DEVICE)

        if DO_TRAIN == True:

            # Create an optimizer object
            optimizer = optimizers[OPTIMIZER](Dunet.parameters(), lr=LEARNING_RATE)

            # Create a criterion object
            criterion = criterion[CRITERION]

            logger.info("-" * 50)
            num_params = sum(p.numel()
                            for p in Dunet.parameters() if p.requires_grad)
        
            logger.info(
                f"Starting training with {DEFENSE_MODEL} that has {num_params} parameters")
            logger.info(f"Learning rate: {LEARNING_RATE}")

            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []

            # Training and evaluating
            for epoch in range(EPOCHS):
                logger.info(f"--- Epoch: {epoch} ---")
                epoch_loss_train, epoch_acc_train = train.train(
                    model=Dunet,
                    loader=train_adv_loader,
                    target_model = resnet,
                    optimizer=optimizer,
                    criterion=criterion,
                    epoch=epoch,
                    epochs=EPOCHS
                )

                epoch_loss_val, epoch_acc_val = val.val(
                    model=Dunet,
                    loader=val_adv_loader,
                    target_model = resnet,
                    optimizer=optimizer,
                    criterion=criterion,
                    epoch=epoch,
                    epochs=EPOCHS
                )

                # Save the losses and accuracies for plotting later
                train_losses.append(epoch_loss_train)
                train_accuracies.append(epoch_acc_train)
                val_losses.append(epoch_loss_val)
                val_accuracies.append(epoch_acc_val)

            # Save the metrics to a JSON file
            results = {
                "train_losses": train_losses,
                "train_accuracies": train_accuracies,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{RESULTS_PATH}/metrics_{timestamp}.json"
            with open(file_name, "w") as f:
                json.dump(results, f)

            logger.info(f"Training and Evaluation finished after {EPOCHS} epochs")

            # Save the model pth and the arquitecture
            savemodel.save_model(Dunet)

            logger.info("-" * 50)

        if DO_TEST == True:

            logger.info("Doing Tests")
        
            # Accuracy of the test set with attacked images
            logger.info("Testing without defense")
            correct = 0
            total = 0

            with torch.no_grad():  
                for adv_img, true_labels, _, predicted_labels_attacked in test_adv_loader:
                    adv_img, true_labels, predicted_labels_attacked = adv_img.to(DEVICE), true_labels.to(DEVICE), predicted_labels_attacked.to(DEVICE)

                    correct += (true_labels == predicted_labels_attacked).sum().item()
                    total += true_labels.size(0)

            accuracy = correct / total if total > 0 else 0
            logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")


            # Accuracy of the test set with a defense in attacked images
            logger.info("Testing with defense")

            # Loading the model
            Dunet.load_state_dict(torch.load(os.path.join(MODELS_PATH,MODEL_SAVED_NAME), map_location=DEVICE))
            Dunet.to(DEVICE)
            Dunet.eval()

            correct = 0
            total = 0

            with torch.no_grad():  
                for adv_img, true_labels, _, predicted_labels_attacked in test_adv_loader:
                    adv_img, true_labels, predicted_labels_attacked = adv_img.to(DEVICE), true_labels.to(DEVICE), predicted_labels_attacked.to(DEVICE)

                    # Count correct predictions
                    outputs = Dunet(adv_img)
                    output_target_model = resnet(outputs)
                    _, predicted = torch.max(output_target_model.data, 1)
                    correct += (true_labels == predicted).sum().item()
                    total += true_labels.size(0)

            accuracy = correct / total if total > 0 else 0
            logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")






        




        



