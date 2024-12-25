import matplotlib.pyplot as plt

def plot_images(original_image, adversarial_image, original_label, predicted_label):
    """
    Plots the original and adversarial images side by side with their respective labels.
    
    Parameters:
    - original_image: The original input image (numpy array or PyTorch tensor).
    - adversarial_image: The adversarially perturbed image (numpy array or PyTorch tensor).
    - original_label: The true label of the image (string or class index).
    - predicted_label: The label predicted after the attack (string or class index).
    """
    # Ensure images are in the range [0, 1] for display
    original_image = original_image.clip(0, 1)
    adversarial_image = adversarial_image.clip(0, 1)
    
    # Create the figure
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.transpose(1, 2, 0) if original_image.ndim == 3 else original_image, cmap='gray')
    plt.title(f"Original Image\nLabel: {original_label}")
    plt.axis('off')
    
    # Plot adversarial image
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_image.transpose(1, 2, 0) if adversarial_image.ndim == 3 else adversarial_image, cmap='gray')
    plt.title(f"Adversarial Image\nPredicted Label: {predicted_label}")
    plt.axis('off')
    
    # Display the images
    plt.tight_layout()
    plt.show()
