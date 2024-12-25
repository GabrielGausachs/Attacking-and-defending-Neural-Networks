import os
import torch

# -----------------------------------------
# Training configuration
# -----------------------------------------

MODELNAME = 'resnet18'
CRITERION = 'CrossEntropyLoss'

# -----------------------------------------
# Main steps
# -----------------------------------------

DO_ATTACK = False
DO_DEFENSE = False
ATTACK_NAME = "ifgsm"

# -----------------------------------------
# Paths 
# -----------------------------------------

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"ILSVRC2012_img_val")
LABEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"val")

# -----------------------------------------
# Parameters 
# -----------------------------------------

BATCH_SIZE_ATTACK = 1
NUM_WORKERS = 2
IMAGES_TO_TEST = 5
EPSILON = 0.5
STEPSIZE = 0.5
NUM_ITERATIONS = 20


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")