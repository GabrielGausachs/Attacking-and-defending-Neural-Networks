import os
import torch

# -----------------------------------------
# Training configuration
# -----------------------------------------

MODELNAME = 'resnet18'

# -----------------------------------------
# Main steps
# -----------------------------------------

DO_ATTACK = False
DO_DEFENSE = False

# -----------------------------------------
# Paths 
# -----------------------------------------

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMAGE_PATH = ""

# -----------------------------------------
# Parameters 
# -----------------------------------------



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")