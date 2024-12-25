# evaluate.py
from data_preprocessing import get_data_generators
from model import create_model
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix
import numpy as np
import csv
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
        self.model = create_model()
        self.load_model()

    def load_model(self):
        """Load the Keras model from the specified path"""
        try:
            self.model.load_weights(self.model_path)
        except Exception as e:
            print(f"Error loading model from {self.model_path}: {str(e)}")
            raise

    def evaluate_model(self):
        model = create_model()
        model.load_weights(self.model_path)
        _, _, test_generator = get_data_generators()
        precision = Precision()
        recall = Recall()
        loss, accuracy = model.evaluate(test_generator)
        y_pred = model.predict(test_generator)
        y_true = test_generator.classes
        y_pred_classes = np.argmax(y_pred, axis=1)
        precision.update_state(y_true, y_pred_classes)
        recall.update_state(y_true, y_pred_classes)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate specificity
        specificity = tn / (tn + fp)
        
        results = {
            "Test Loss": loss,
            "Test Accuracy": accuracy,
            "Precision": precision.result().numpy(),
            "Recall": recall.result().numpy(),
            "Specificity": specificity
        }
        csv_file_path = os.path.join("models", "evaluation_results.csv")
        file_exists = os.path.exists(csv_file_path)
        with open(csv_file_path, "a", newline="") as csvfile:
            fieldnames = results.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
        print(f"Test Loss: {results['Test Loss']:.4f}")
        print(f"Test Accuracy: {results['Test Accuracy']:.4f}")
        print(f"Precision: {results['Precision']:.4f}")
        print(f"Recall: {results['Recall']:.4f}")
        print(f"Specificity: {results['Specificity']:.4f}")
        print(f"Results saved to: {os.path.abspath(csv_file_path)}")


if __name__ == "__main__":
    model_path = "models/trained_model.keras"
    predictor = ImagePredictor(model_path)
    predictor.evaluate_model()
