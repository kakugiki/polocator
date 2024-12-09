from data_preprocessing import get_data_generators
from model import create_model
import os


def train_model():
    model = create_model()
    train_generator, val_generator, num_classes = get_data_generators()
    model_path = os.path.join("models", "trained_model.h5")

    # Training logic using model.fit
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,  # Adjust as needed
        validation_data=val_generator,
        validation_steps=len(val_generator),
        verbose=1,
    )

    model.save(model_path)
    return model


if __name__ == "__main__":
    trained_model = train_model()
    print(f"Model saved to: {os.path.abspath('models/trained_model.h5')}")
