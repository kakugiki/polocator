# train.py
from data_preprocessing import get_data_generators
from model import create_model

def train_model():
    model = create_model()
    train_generator, val_generator, _ = get_data_generators()
    # ... training logic
    return model

if __name__ == "__main__":
    trained_model = train_model()
    # Save the model