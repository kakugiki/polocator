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
        Initialize the Predict class with model path and target size

        Args:
            model_path (str): Path to the trained model
            target_size (tuple): Target size for input images (width, height)
        """
        self.model_path = model_path
        self.target_size = target_size
        self.model = self._load_model()

    def _load_model(self):
        """Load the model from the specified path"""
        try:
            return load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {str(e)}")
            raise

    @staticmethod
    def load_image(image_path, target_size):
        """
        Load and convert image to RGB format, handling both HEIC and regular formats

        Args:
            image_path (str): Path to the image
            target_size (tuple): Desired output size (width, height)

        Returns:
            PIL.Image: Converted and resized image
        """
        if image_path.lower().endswith(".heic"):
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
            image = Image.open(image_path)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image
        image = image.resize(target_size)
        return image

    def process_folder(self, folder_path, output_csv="models/prediction_results.csv"):
        """
        Process all images in a folder and save results to CSV

        Args:
            folder_path (str): Path to folder containing images
            output_csv (str): Path where to save the CSV results
        """
        results = []
        filenames = []

        # Process each image in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith((".png", ".jpg", ".jpeg", ".heic")):
                img_path = os.path.join(folder_path, filename)

                try:
                    # Load and preprocess image
                    img = self.load_image(img_path, target_size=self.target_size)
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0

                    # Predict
                    prediction = self.model.predict(img_array, verbose=0)
                    class_probabilities = prediction[0]

                    # Store results
                    results.append(class_probabilities)
                    filenames.append(filename)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

        # Create and save DataFrame
        if not results:
            print("No results found. Check if any valid images were processed.")
            return None

        print(f"Number of files processed: {len(filenames)}")
        print(f"Shape of first result: {results[0].shape if results else 'No results'}")

        df = pd.DataFrame(
            results,
            index=filenames,
            columns=[f"Class_{i}" for i in range(len(results[0]))],
        )
        df["Predicted_Class"] = df.apply(lambda row: np.argmax(row), axis=1)

        # Save results
        df.to_csv(output_csv)
        return df

    def predict_single(self, image_path):
        """
        Make prediction for a single image

        Args:
            image_path (str): Path to the image file

        Returns:
            tuple: (predicted_class, class_probabilities)
        """
        img = self.load_image(image_path, self.target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return predicted_class, prediction[0]


# Example usage
if __name__ == "__main__":
    predictor = ImagePredictor()

    # Process entire folder
    results_df = predictor.process_folder(
        folder_path="data/processed/test/control",
        output_csv="models/prediction_results.csv",
    )

    # Or make single prediction
    # image_path = "data/processed/test/control/C_023.jpg"
    # class_pred, probabilities = predictor.predict_single(image_path)
    # print(f"Predicted class: {class_pred}")
    # print(f"Class probabilities: {probabilities}")
