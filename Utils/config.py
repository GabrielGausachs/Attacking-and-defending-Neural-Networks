import os
import torch

# -----------------------------------------
# Training configuration
# -----------------------------------------

MODELNAME = 'resnet18'
CRITERION = 'CrossEntropyLoss'
OPTIMIZER = "Adam"
DEFENSE_MODEL = "DUNet"


# -----------------------------------------
# Main steps
# -----------------------------------------

DO_ATTACK = True
DO_DEFENSE = False
ATTACK_NAME = "ifgsm"

# -----------------------------------------
# Paths 
# -----------------------------------------

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
IMAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"ILSVRC2012_img_val")
NUM_LABELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),"ILSVRC2012_bbox_val_v3/val")
LABELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Files/imagenet_class_index.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Results")
ADV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Adversarial_images")

# -----------------------------------------
# Parameters 
# -----------------------------------------

BATCH_SIZE_ATTACK = 16
BATCH_SIZE_UNET = 32
NUM_WORKERS = 5
IMAGES_TO_TEST = 16
EPSILON = 0.03
STEPSIZE = 0.005
NUM_ITERATIONS = 10
LEARNING_RATE = 0.01
EPOCHS = 25


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")