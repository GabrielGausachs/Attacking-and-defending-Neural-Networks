import matplotlib.pyplot as plt
import torch
from datetime import datetime


from Utils.config import ( 
    RESULTS_PATH,
)

from Utils.logger import initialize_logger, get_logger



logger = get_logger()

def plot_images(original_image, adversarial_image, original_label, predicted_label):
    """
    Plots the original and adversarial images side by side with their respective labels.
    
    Parameters:
    - original_image: The original input image (numpy array or PyTorch tensor).
    - adversarial_image: The adversarially perturbed image (numpy array or PyTorch tensor).
    - original_label: The true label of the image (string or class index).
    - predicted_label: The label predicted after the attack (string or class index).
    """

    # Ensure images are detached from computation graph and on CPU
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.squeeze().detach().cpu().numpy()  # Remove batch dimension
    if isinstance(adversarial_image, torch.Tensor):
        adversarial_image = adversarial_image.squeeze().detach().cpu().numpy()  # Remove batch dimension
    
    # Transpose if the image is in (C, H, W) format
    if original_image.ndim == 3 and original_image.shape[0] == 3:  # RGB image
        original_image = original_image.transpose(1, 2, 0)
    if adversarial_image.ndim == 3 and adversarial_image.shape[0] == 3:  # RGB image
        adversarial_image = adversarial_image.transpose(1, 2, 0)
    
    # Ensure images are in the range [0, 1] for display
    original_image = original_image.clip(0, 1)
    adversarial_image = adversarial_image.clip(0, 1)

    # Path to save the figure
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path_figure = f"{RESULTS_PATH}/{current_date}_label_{original_label}.png"
    
    # Create the figure
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray' if original_image.ndim == 2 else None)
    plt.title(f"Original Image\nLabel: {original_label}")
    plt.axis('off')
    
    # Plot adversarial image
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_image, cmap='gray' if adversarial_image.ndim == 2 else None)
    plt.title(f"Adversarial Image\nPredicted Label: {predicted_label}")
    plt.axis('off')
    
    # Display the images
    plt.tight_layout()
    plt.savefig(save_path_figure, bbox_inches='tight')
    logger.info("-" * 50)
    logger.info(f"Saved figure to {save_path_figure}")
    logger.info("-" * 50)
    plt.show()
