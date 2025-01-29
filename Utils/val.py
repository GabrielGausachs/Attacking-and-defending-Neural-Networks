from Utils.logger import initialize_logger, get_logger
import torch
import gc

from Utils.config import (
    DEVICE,
)

logger = get_logger()


def val(model, target_model, loader, optimizer, criterion, epoch=0, epochs=0):
    total_loss = 0
    total = 0
    correct = 0

    model.eval()

    logger.info(f"Epoch: {epoch}/{epochs}, Starting validation...")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Memory cleaned!")

    with torch.no_grad():
        for batch_idx, (adv_images, true_label, pred_label) in enumerate(loader, 1):
            adv_images = adv_images.to(DEVICE)
            true_label = true_label.to(DEVICE)
            pred_label = pred_label.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(adv_images)
            output_target_model = target_model(outputs)
            train_loss = criterion(output_target_model, true_label)
            logger.info(f"Batch_idx: {batch_idx} - Validation loss = {train_loss:.6f}")

            total_loss += train_loss.item()
            # print('loss:',train_loss.item())

            # Calculate accuracy
            _, predicted = torch.max(output_target_model.data, 1)
            total += true_label.size(0)
            correct += (predicted == true_label).sum().item()

            # Free memory in each iteration
            del adv_images
            del true_label
            del pred_label
            del train_loss
            torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
            gc.collect()  # Collect trash to free memory not used


    epoch_loss = total_loss / len(loader)
    print(epoch_loss)
    accuracy = 100 * correct / total

    logger.info(f"Epoch: {epoch}/{epochs}, Validation loss = {epoch_loss:.6f},  Accuracy = {accuracy:.2f}%")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Evaluation finished! Memory cleaned!")
    logger.info("-" * 50)

    return epoch_loss, accuracy