from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
from PIL import Image
import pillow_heif
import os


class ImagePredictor:
    def __init__(self, model_path="models/best_model.keras", target_size=(224, 224)):
        """
        Initialize the ImagePredictor with a model and target image size.

        Args:
            model_path (str): Path to the trained Keras model
            target_size (tuple): Target size for input images (width, height)
        """
        self.model_path = model_path
        self.target_size = target_size
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the Keras model from the specified path"""
        try:
            self.model = load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {str(e)}")
            raise

    def load_image(self, image_path):
        """
        Load and resize images, supporting both HEIC and common formats (JPG, JPEG, PNG)

        Args:
            image_path (str): Path to the image file

        Returns:
            PIL.Image: Converted and resized image
        """
        # Get file extension (lowercase)
        file_ext = os.path.splitext(image_path)[1].lower()

        try:
            if file_ext in [".heic", ".heif"]:
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
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize image
            image = image.resize(self.target_size)
            return image

        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            raise

    def preprocess_image(self, image):
        """
        Preprocess the image for model prediction

        Args:
            image (PIL.Image): Input image

        Returns:
            numpy.ndarray: Preprocessed image array
        """
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict(self, image_path):
        """
        Make predictions on a single image

        Args:
            image_path (str): Path to the image file

        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        # Load and preprocess the image
        img = self.load_image(image_path)
        img_array = self.preprocess_image(img)

        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return predicted_class, predictions[0]

    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images

        Args:
            image_paths (list): List of paths to image files

        Returns:
            list: List of tuples containing (predicted_class, class_probabilities)
        """
        results = []
        for image_path in image_paths:
            try:
                predicted_class, probabilities = self.predict(image_path)
                results.append(
                    {
                        "image_path": image_path,
                        "predicted_class": predicted_class,
                        "probabilities": probabilities,
                    }
                )
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = ImagePredictor()

    # Single image prediction
    img_path = "data/processed/test/control/C_023.jpg"
    predicted_class, probabilities = predictor.predict(img_path)
    print(f"Predicted class: {predicted_class}")
    print(f"Class probabilities: {probabilities}")
