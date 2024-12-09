from data_preprocessing import get_data_generators
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


def train_model():
    model = create_model()
    train_generator, val_generator, num_classes = get_data_generators()
    model_path = os.path.join("models", "trained_model.keras")

    # Early stopping to halt training when the validation loss does not improve
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Save the best model based on validation loss with .keras extension
    best_model_checkpoint = ModelCheckpoint(
        "models/best_model.keras", monitor="val_loss", save_best_only=True, mode="min"
    )

    # Save the last model weights of each epoch with .weights.h5 extension
    last_model_checkpoint = ModelCheckpoint(
        "models/last_model.weights.h5",  # Weights-only save can use .h5
        save_weights_only=True,  # Only saving weights
        save_freq="epoch",
    )

    # Combine callbacks into a list to pass to model.fit
    callbacks_list = [early_stopping, best_model_checkpoint, last_model_checkpoint]

    # Training logic using model.fit
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,  # Now you can set a higher number knowing early stopping will manage it
        validation_data=val_generator,
        validation_steps=len(val_generator),
        verbose=1,
        callbacks=[callbacks_list],
    )

    model.save(model_path)
    return model


if __name__ == "__main__":
    trained_model = train_model()
    print(f"Model saved to: {os.path.abspath('models/trained_model.keras')}")
