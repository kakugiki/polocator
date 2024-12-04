import tensorflow as tf
import os
import cv2
import numpy as np
import re

# Directory containing original and where augmented images will be saved
image_dir = "data/processed/dog_poop"


# Function to find the highest number in image names
def find_highest_image_number(directory):
    max_num = 0
    for file in os.listdir(directory):
        if file.endswith((".png", ".jpg", ".jpeg")):
            match = re.search(r"(\w+)_(\d+)\.", file)
            if match:
                num = int(match.group(2))
                if num > max_num:
                    max_num = num
    return max_num


# Find the highest number in original images
last_original_num = find_highest_image_number(image_dir)

# Correctly calculate the next number, handling padding
next_num = last_original_num + 1
next_num_str = "{:03d}".format(next_num)


# Load images
def load_images(path):
    images = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(path, filename)
            image = cv2.imread(img_path)
            if image is not None:
                images.append((image, filename))  # Keep filename for reference
    return images


# Load and sort images
X = load_images(image_dir)

# Define augmentation using tf.keras.layers
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
    ]
)

# Augment and save images
for img, original_filename in X:
    # Generate augmented images
    augmented_images = []
    for i in range(5):
        augmented_image = data_augmentation(np.expand_dims(img, axis=0))
        augmented_images.append(augmented_image[0].numpy())

    for augmented_image in augmented_images:
        # Extract the base name without the number
        base_name = original_filename.split("_")[0]
        # Save with incremented number and padding
        new_filename = f"{base_name}_{next_num_str}.jpg"
        cv2.imwrite(
            os.path.join(image_dir, new_filename), augmented_image.astype(np.uint8)
        )
        next_num += 1
        next_num_str = "{:03d}".format(next_num)
