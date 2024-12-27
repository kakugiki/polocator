# evaluate.py
from data_preprocessing import get_data_generators
from model import create_model
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix
import numpy as np
import csv
import os


class ImageEvaluator:
    def __init__(self, model_path="models/best_model.keras", target_size=(100, 100)):
        """
        Initialize the ImageEvaluator with a model and target image size.

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
        # Evaluate model
        loss, accuracy = model.evaluate(test_generator)

        # Get predictions
        y_pred = model.predict(test_generator)
        y_true = test_generator.classes
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        # Calculate metrics for each class
        class_names = list(test_generator.class_indices.keys())
        metrics_per_class = {}

        for i, class_name in enumerate(class_names):
            # Convert to binary classification for each class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred_classes == i).astype(int)

            # Calculate metrics
            class_cm = confusion_matrix(y_true_binary, y_pred_binary)
            tn, fp, fn, tp = class_cm.ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            metrics_per_class[class_name] = {
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity,
            }

        # Compute average metrics
        avg_precision = np.mean([m["Precision"] for m in metrics_per_class.values()])
        avg_recall = np.mean([m["Recall"] for m in metrics_per_class.values()])
        avg_specificity = np.mean(
            [m["Specificity"] for m in metrics_per_class.values()]
        )

        results = {
            "Test Loss": loss,
            "Test Accuracy": accuracy,
            "Average Precision": avg_precision,
            "Average Recall": avg_recall,
            "Average Specificity": avg_specificity,
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
        print(f"Average Precision: {results['Average Precision']:.4f}")
        print(f"Average Recall: {results['Average Recall']:.4f}")
        print(f"Average Specificity: {results['Average Specificity']:.4f}")

        # Print per-class metrics
        print("\nPer-class metrics:")
        for class_name, metrics in metrics_per_class.items():
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall: {metrics['Recall']:.4f}")
            print(f"  Specificity: {metrics['Specificity']:.4f}")
        print(f"Results saved to: {os.path.abspath(csv_file_path)}")


if __name__ == "__main__":
    model_path = "models/resnet50_model.keras"
    evaluator = ImageEvaluator(model_path)
    evaluator.evaluate_model()
