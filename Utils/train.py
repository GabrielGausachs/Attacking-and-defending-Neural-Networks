from Utils.logger import initialize_logger, get_logger
import torch
import gc

from Utils.config import (
    DEVICE,
)

logger = get_logger()


def train(model, target_model, loader, optimizer, criterion, epoch=0, epochs=0):
    total_loss = 0
    total_metric = [0, 0, 0, 0]
    model.train()

    logger.info(f"Epoch: {epoch}/{epochs}, Starting training...")

    # Logger info
    logger.info(f"Loader length: {len(loader)}")
    logger.info(f"Loader batch size: {loader.batch_size}")
    logger.info(f"Loader drop last: {loader.drop_last}")
    logger.info(f"Loader num workers: {loader.num_workers}")
    logger.info(f"Criterion: {criterion}")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Memory cleaned!")

    for batch_idx, (adv_images, true_label, pred_label) in enumerate(loader, 1):
        adv_images = adv_images.to(DEVICE)
        true_label = true_label.to(DEVICE)
        print(adv_images.size())

        optimizer.zero_grad()
        outputs = model(adv_images)
        logger.info(f"Output DUNet: {outputs.size()}")
        output_target_model = target_model(outputs)
        logger.info(f"Output target model: {output_target_model.size()} with value: {output_target_model}")
        train_loss = criterion(output_target_model, true_label)

        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()
        # print('loss:',train_loss.item())

        # Free memory in each iteration
        del adv_images
        del true_label
        del pred_label
        del train_loss
        torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
        gc.collect()  # Collect trash to free memory not used


    epoch_loss = total_loss / len(loader)
    print(epoch_loss)

    logger.info(f"Epoch: {epoch}/{epochs}, Train loss = {epoch_loss:.6f}")

    torch.cuda.empty_cache()  # Clean CUDA Cache if used GPU
    gc.collect()  # Collect trash to free memory not used
    logger.info("Train finished! Memory cleaned!")
    logger.info("-" * 50)

    return epoch_loss