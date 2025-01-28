import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import os
import pandas as pd
import json
import numpy as np
from PIL import Image


from Utils.config import ( 
    RESULTS_PATH,
    ADV_PATH,
    LABELS_PATH,
)

from Utils.logger import initialize_logger, get_logger



logger = get_logger()


excel_file = os.path.join(ADV_PATH, "metadata.xlsx")  
df = pd.read_excel(excel_file)


# Subset where Predicted_Label_Previous != Predicted_Label
df_diff = df[df['Predicted_Label_Previous'] != df['Predicted_Label']]

# Subset where Predicted_Label_Previous == Predicted_Label
df_same = df[df['Predicted_Label_Previous'] == df['Predicted_Label']]

# Randomly select one row from each subset
random_row_diff = df_diff.sample(n=1).iloc[0] if not df_diff.empty else None
random_row_same = df_same.sample(n=1).iloc[0] if not df_same.empty else None

    
# Select a random row
def get_images(random_row):
    imagename = random_row['Image_Name']
    true_label = int(random_row['True_Label'])
    predicted_label_previous = int(random_row['Predicted_Label_Previous'])
    predicted_label = int(random_row['Predicted_Label'])
        
    # Load the JSON file from LABELS_PATH
    json_file = os.path.join(LABELS_PATH)  # Replace 'labels.json' with your actual file name
    with open(json_file, 'r') as f:
        label_map = json.load(f)
        
    # Map numeric labels to class names
    original_label_name = label_map[str(true_label)][1]
    predicted_label_previous_name = label_map[str(predicted_label_previous)][1]
    predicted_label_name = label_map[str(predicted_label)][1]

    print(original_label_name)
    print(imagename)
    adversarial_image_path = os.path.join(ADV_PATH, imagename)
    adversarial_image = Image.open(adversarial_image_path)

    plot_images(adversarial_image,original_label_name,predicted_label_previous_name,predicted_label_name)


def plot_images(adversarial_image, original_label, predicted_label_previous, predicted_label):

    if isinstance(adversarial_image, Image.Image):
        adversarial_image = np.array(adversarial_image) / 255.0  # Normalize to [0, 1]
    
    # Ensure images are in the range [0, 1]
    adversarial_image = adversarial_image.clip(0, 1)

    #plt.imsave(f"ad_{original_label}.png", adversarial_image)

    # Path to save the figure
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path_figure = f"{RESULTS_PATH}/{current_date}_label_{original_label}.png"
    
    # Create the figure
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(adversarial_image, cmap='gray' if adversarial_image.ndim == 2 else None)
    plt.title(f"Original Image\nTrue Label: {original_label}\nBefore Attacking: {predicted_label_previous}")
    plt.axis('off')
    
    # Plot adversarial image
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial_image, cmap='gray' if adversarial_image.ndim == 2 else None)
    plt.title(f"Adversarial Image\After Attacking: {predicted_label}")
    plt.axis('off')
    
    # Display the images
    plt.tight_layout()
    plt.savefig(save_path_figure, bbox_inches='tight')
    logger.info("-" * 50)
    logger.info(f"Saved figure to {save_path_figure}")
    logger.info("-" * 50)
    plt.show()

get_images(random_row_diff)
get_images(random_row_same)
