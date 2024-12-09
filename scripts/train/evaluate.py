# evaluate.py
from data_preprocessing import get_data_generators
from model import create_model
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
import csv
import os

def evaluate_model(model_path):
    model = create_model()
    model.load_weights(model_path)
    _, _, test_generator = get_data_generators()
    precision = Precision()
    recall = Recall()
    loss, accuracy = model.evaluate(test_generator)
    y_pred = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred_classes = np.argmax(y_pred, axis=1)
    precision.update_state(y_true, y_pred_classes)
    recall.update_state(y_true, y_pred_classes)
    results = {
        "Test Loss": loss,
        "Test Accuracy": accuracy,
        "Precision": precision.result().numpy(),
        "Recall": recall.result().numpy(),
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
    print(f"Results saved to: {os.path.abspath(csv_file_path)}")


if __name__ == "__main__":
    model_path = "models/trained_model.h5"
    evaluate_model(model_path)
