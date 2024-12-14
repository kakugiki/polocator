from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
from PIL import Image
import pillow_heif
import os


def load_image(image_path, target_size):
    """
    Load and resize images, supporting both HEIC and common formats (JPG, JPEG, PNG)
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Desired output size (width, height)
        
    Returns:
        PIL.Image: Converted and resized image
    """
    # Get file extension (lowercase)
    file_ext = os.path.splitext(image_path)[1].lower()
    
    try:
        if file_ext in ['.heic', '.heif']:
            # Handle HEIC/HEIF format
            heif_file = pillow_heif.read_heif(image_path)
            image = Image.frombytes(
                heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        else:
            # Handle other image formats (JPG, JPEG, PNG, etc.)
            image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size)
        return image
        
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        raise


model = load_model("models/best_model.keras")


img_path = "data/raw/control/2024-09-13_15-02-26_886.heic"
img = load_image(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class[0]}")
print(f"Class probabilities: {predictions[0]}")
