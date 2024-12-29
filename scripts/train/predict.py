from model import create_model
from data_preprocessing import get_data_generators
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import pandas as pd
from PIL import Image
import pillow_heif
import os

import matplotlib.pyplot as plt
import seaborn as sns


class ImagePredictor:
    def __init__(self, model_path="models/resnet50_model.keras", target_size=(100, 100)):
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
            model = create_model()
            model.load_weights(self.model_path)
            return model
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

        # Get class names from data generator
        _, _, test_generator = get_data_generators()
        class_names = list(test_generator.class_indices.keys())

        # Process each image in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith((".png", ".jpg", ".jpeg", ".heic")):
                img_path = os.path.join(folder_path, filename)

                try:
                    # Load and preprocess image
                    img = self.load_image(img_path, target_size=self.target_size)
                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(
                        img_array
                    )  # Use ResNet50's preprocessing

                    # Predict
                    prediction = self.model.predict(img_array, verbose=0)

                    # Store results (probabilities for all classes)
                    results.append(prediction[0])
                    filenames.append(filename)

                    # Debugging: Log intermediate values
                    predicted_class = np.argmax(prediction[0])
                    class_probabilities = {
                        class_names[i]: prob for i, prob in enumerate(prediction[0])
                    }
                    # print(f"Processed {filename}:")
                    # print(f"  Predicted class: {class_names[predicted_class]}")
                    # print(f"  Class probabilities: {class_probabilities}")

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

        # Create and save DataFrame
        if not results:
            print("No results found. Check if any valid images were processed.")
            return None

        print(f"Number of files processed: {len(filenames)}")

        # Create DataFrame with probabilities for each class
        df = pd.DataFrame(
            results,
            index=filenames,
            columns=[f"{name}_Probability" for name in class_names],
        )

        # Add predicted class column
        df["Predicted_Class"] = df.idxmax(axis=1).apply(
            lambda x: x.replace("_Probability", "")
        )

        # Save results
        df.to_csv(output_csv)
        return df

    def predict_single(self, image_path):
        """
        Make prediction for a single image

        Args:
            image_path (str): Path to the image file

        Returns:
            tuple: (predicted_class_name, class_probabilities)
        """
        # Get class names from data generator
        _, _, test_generator = get_data_generators()
        class_names = list(test_generator.class_indices.keys())

        img = self.load_image(image_path, self.target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Use ResNet50's preprocessing

        prediction = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        class_probabilities = {
            class_names[i]: prob for i, prob in enumerate(prediction[0])
        }

        # Debugging: Log intermediate values
        print(f"Processed {image_path}:")
        print(f"  Predicted class: {class_names[predicted_class_idx]}")
        print(f"  Class probabilities: {class_probabilities}")

        return class_names[predicted_class_idx], class_probabilities

    def plot_prediction_results(self, df):
        """
        Create visualizations for prediction results
        
        Args:
            df: DataFrame containing prediction results
        """
        plt.figure(figsize=(12, 6))

        # 1. Bar plot of class distribution
        plt.subplot(1, 2, 1)
        class_counts = df['Predicted_Class'].value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title('Distribution of Predicted Classes')
        plt.xticks(rotation=45)
        plt.ylabel('Count')

        # 2. Box plot of prediction probabilities
        plt.subplot(1, 2, 2)
        prob_columns = [col for col in df.columns if col.endswith('_Probability')]
        prob_data = df[prob_columns]
        prob_data.columns = [col.replace('_Probability', '') for col in prob_columns]
        sns.boxplot(data=prob_data)
        plt.title('Distribution of Class Probabilities')
        plt.xticks(rotation=45)
        plt.ylabel('Probability')

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    model_path = "models/resnet50_model.keras"
    predictor = ImagePredictor(model_path)

    # Process test folders
    print("\nProcessing control images:")
    control_results = predictor.process_folder(
        folder_path="data/processed/test/control",
        output_csv="models/prediction_results_control.csv",
    )

    print("\nProcessing positive images:")
    positive_results = predictor.process_folder(
        folder_path="data/processed/test/positive",
        output_csv="models/prediction_results_positive.csv",
    )
    if control_results is not None:
        predictor.plot_prediction_results(control_results)
    if positive_results is not None:
        predictor.plot_prediction_results(positive_results)

    # # Example of single prediction
    # print("\nSingle image prediction example:")
    # image_path = "data/processed/test/control/C_023.jpg"
    # predicted_class, probabilities = predictor.predict_single(image_path)
    # print(f"\nFinal prediction for {image_path}:")
    # print(f"  Predicted class: {predicted_class}")
    # for class_name, prob in probabilities.items():
    #     print(f"  {class_name} probability: {prob:.4f}")
