from Utils.logger import initialize_logger,get_logger
import torch
import datetime

from Utils.config import (
    DEFENSE_MODEL,
    MODELS_PATH,
)


logger = get_logger()

def save_model(model):
    
    # Save the model pth and the arquitecture
    logger.info("Saving the model")
    model.to("cpu")
    torch.save(
        model.state_dict(), f"{MODELS_PATH}/DUNET_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
    )
    logger.info(f"Model saved in {MODELS_PATH}")