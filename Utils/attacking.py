import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from Utils.config import ( 
    DEVICE,
)

from Utils.logger import initialize_logger, get_logger



logger = get_logger()


def I_FGM_attack(model, image, label, criterion, epsilon, alpha, num_iter):
    
    """
    Perform the Iterative FGSM (I-FGSM) attack on a single image.

    :param model: The neural network model
    :param image: The original input image tensor
    :param label: The true label of the image
    :param epsilon: The maximum perturbation allowed
    :param alpha: The step size for each iteration
    :param num_iter: The number of iterations for the attack
    :param device: The device (CPU or cuda)
    :return: The adversarial example
    """

    # Set model to evaluation
    model.eval()

    # Move image to the correct device
    image = image.to(DEVICE)
    label = label.to(DEVICE)

    # Create a copy of the image to modify during the attack
    image_adv = image.clone().detach().requires_grad_(True)

    # Iterating to apply perturbation
    for t in range(num_iter):

        logger.info(f"Iteration {t + 1}/{num_iter}")

        output = model(image_adv)
        loss = criterion(output, label)

        pred_label = output.argmax(dim=1).item()
        logger.info(f"Current predicted label: {pred_label}, Loss: {loss.item():.4f}")

        # Zero all previous gradients
        model.zero_grad()

        # Backward pass: Compute gradients
        loss.backward()

        # Get the sign of the gradients
        grad_sign = image_adv.grad.sign()

        # Apply the perturbation: update the image
        image_adv = image_adv + alpha * grad_sign

        # Clip the perturbation to ensure it's within the allowed epsilon ball
        perturbation = torch.clamp(image_adv - image, min=-epsilon, max=epsilon)
        image_adv = torch.clamp(image + perturbation, 0, 1)

        logger.info(f"Updated adversarial image with perturbation norm: {perturbation.norm().item():.4f}")

        # Re-zero the gradients after each update
        image_adv.grad.zero_()
    
    output = model(image_adv)
    pred_label = output.argmax(dim=1).item()
    logger.info(f"Label predicted after attacking: {pred_label}, Loss: {loss.item():.4f}")


    return image_adv, pred_label
