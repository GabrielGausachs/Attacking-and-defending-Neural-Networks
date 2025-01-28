from Utils.logger import initialize_logger, get_logger
import torch
import gc

from Utils.config import (
    DEVICE,
)

logger = get_logger()


def train(model, target_model, loader, optimizer, criterion, epoch=0, epochs=0):
    total_loss = 0
    model.train()

    logger.info(f"Epoch: {epoch}/{epochs}, Starting training...")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Memory cleaned!")

    for batch_idx, (adv_images, true_label, pred_label) in enumerate(loader, 1):
        adv_images = adv_images.to(DEVICE)
        true_label = true_label.to(DEVICE)
        print(adv_images.size())

        optimizer.zero_grad()
        outputs = model(adv_images)
        #logger.info(f"Output DUNet: {outputs.size()}")
        output_target_model = target_model(outputs)
        #logger.info(f"Output target model: {output_target_model.size()} with value: {output_target_model}")
        train_loss = criterion(output_target_model, true_label)
        logger.info(f"Batch_idx: {batch_idx} - Train loss = {train_loss:.6f}")

        train_loss.backward()
        optimizer.step()

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
    accuracy = 100 * correct / total

    logger.info(f"Epoch: {epoch}/{epochs}, Train loss = {epoch_loss:.6f}, Accuracy = {accuracy:.2f}%")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Train finished! Memory cleaned!")
    logger.info("-" * 50)

    return epoch_loss, accuracy