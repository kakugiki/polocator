# evaluate.py
from data_preprocessing import get_data_generators
from model import create_model
from tensorflow.keras.metrics import Precision, Recall

def evaluate_model(model_path):
    model = create_model()
    model.load_weights(model_path)
    _, _, test_generator = get_data_generators()
    precision = Precision()
    recall = Recall()
    loss, accuracy = model.evaluate(test_generator)
    y_pred = model.predict(test_generator)
    y_true = test_generator.classes
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision.result().numpy():.4f}")
    print(f"Recall: {recall.result().numpy():.4f}")
