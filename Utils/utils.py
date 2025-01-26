from PIL import Image 
import os
import csv

from Utils.logger import initialize_logger, get_logger

logger = get_logger()

# Python file with functions 

def save_adversial_images(adversial_images, labels, pred_labels, output_folder,metadata_file, image_count):
                
    # Convert adversarial image tensor to PIL image and save it
    print(adversial_images.size(0))
    for idx in range(adversial_images.size(0)):  # Handle batch of images
        single_image = adversial_images[idx]  # Extract one image
        single_image = single_image.squeeze(0)  # Remove batch dimension if needed
        single_image = single_image.cpu().detach()  # Ensure on CPU
        single_label = labels[idx].item()
        single_pred = pred_labels[idx].item()
        single_image = single_image.permute(1, 2, 0)  # Rearrange to [H, W, C]
        single_image = (single_image * 255).byte().numpy()  # Scale to [0, 255]

        pil_image = Image.fromarray(single_image)  # Convert to PIL Image
        image_name = f"adversarial_image_{image_count}.png"
        image_path = os.path.join(output_folder, image_name)
        pil_image.save(image_path) # Save as PNG
        logger.info(f"Saved adversarial image to: {image_path}")

        with open(metadata_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([image_name, single_label, single_pred])
                    
        image_count += 1

        return image_count
