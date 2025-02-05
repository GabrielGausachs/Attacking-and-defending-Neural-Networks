from PIL import Image 
import os
import csv

from Utils.logger import initialize_logger, get_logger

from Utils.config import (
    LABELS_PATH,
)

logger = get_logger()

# ------------------- Python file with functions -------------------

def save_adversial_images(adversial_images, labels, pred_labels, pred_label_previous, output_folder, new_rows, image_count):
                
    """Convert adversarial images tensor to PIL images and save them"""
    print(adversial_images.size(0))
    for idx in range(adversial_images.size(0)):  
        single_image = adversial_images[idx]  
        single_image = single_image.squeeze(0)  
        single_image = single_image.cpu().detach()  
        single_label = labels[idx].item()
        single_pred_label_previous = pred_label_previous[idx].item()
        single_pred = pred_labels[idx].item()
        single_image = single_image.permute(1, 2, 0)  
        single_image = (single_image * 255).byte().numpy()

        pil_image = Image.fromarray(single_image)  # Convert to PIL Image
        image_name = f"adversarial_image_{image_count}.png"
        image_path = os.path.join(output_folder, image_name)
        pil_image.save(image_path) # Save as PNG
        logger.info(f"Saved adversarial image to: {image_path}")



        new_rows.append({"Image_Name": image_name, "True_Label": single_label,"Predicted_Label_Previous":single_pred_label_previous, "Predicted_Label": single_pred})
                    
        image_count += 1

    return image_count,new_rows
