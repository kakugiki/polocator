from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import pillow_heif


def load_heic_image(image_path, target_size):
    """
    Load and convert HEIC image to RGB format

    Args:
        image_path (str): Path to the HEIC image
        target_size (tuple): Desired output size (width, height)

    Returns:
        PIL.Image: Converted and resized image
    """
    heif_file = pillow_heif.read_heif(image_path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image
    image = image.resize(target_size)
    return image


model = load_model("models/best_model.keras")

img_path = "data/raw/2024-09-13_15-03-16_044.heic"
img = load_heic_image(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)
print(f"Predicted class: {predicted_class[0]}")
print(f"Class probabilities: {predictions[0]}")
