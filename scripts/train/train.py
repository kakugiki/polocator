from data_preprocessing import get_data_generators
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os


class ModelTrainer:
    def __init__(self, model_dir="models"):
        """
        Initialize the ModelTrainer with configuration.

        Args:
            model_dir (str): Directory where models will be saved
        """
        self.model_dir = model_dir
        self.model = None
        self.train_generator = None
        self.val_generator = None
        self.num_classes = None

        # Create models directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def setup_model(self):
        """Set up the model and data generators"""
        self.model = create_model()
        self.train_generator, self.val_generator, self.num_classes = (
            get_data_generators()
        )

    def _create_callbacks(self):
        """Create and return the callback list for model training"""
        # Early stopping to halt training when the validation loss does not improve
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        # Save the best model based on validation loss with .keras extension
        best_model_checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )

        # Save the last model weights of each epoch with .weights.h5 extension
        last_model_checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, "last_model.weights.h5"),
            save_weights_only=True,
            save_freq="epoch",
        )

        return [early_stopping, best_model_checkpoint, last_model_checkpoint]

    def train_model(self, initial_epochs=10, fine_tune_epochs=40, verbose=1):
        """
        Train the model using a two-phase approach:
        1. Train only the top layers
        2. Fine-tune the top layers of the base model

        Args:
            initial_epochs (int): Number of epochs for initial training
            fine_tune_epochs (int): Number of epochs for fine-tuning
            verbose (int): Verbosity mode (0, 1, or 2)

        Returns:
            tuple: (initial_history, fine_tune_history)
        """
        if self.model is None:
            self.setup_model()

        callbacks_list = self._create_callbacks()

        # Phase 1: Training only the top layers
        print("Phase 1: Training top layers...")
        initial_history = self.model.fit(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=initial_epochs,
            validation_data=self.val_generator,
            validation_steps=len(self.val_generator),
            verbose=verbose,
            callbacks=callbacks_list,
        )

        # Phase 2: Fine-tuning
        print("Phase 2: Fine-tuning top layers of ResNet50...")
        # Unfreeze the last 30 layers of the ResNet50 base
        base_model = self.model.layers[1]  # ResNet50 is the second layer
        for layer in base_model._layers[-30:]:
            layer.trainable = True

        # Recompile with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(
                learning_rate=1e-5
            ),  # Even lower learning rate for fine-tuning
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        fine_tune_history = self.model.fit(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=fine_tune_epochs,
            validation_data=self.val_generator,
            validation_steps=len(self.val_generator),
            verbose=verbose,
            callbacks=callbacks_list,
        )

        # Save the final model
        model_path = os.path.join(self.model_dir, "trained_model.keras")
        self.model.save(model_path)
        print(f"Model saved to: {os.path.abspath(model_path)}")

        return initial_history, fine_tune_history

    def get_model(self):
        """Return the trained model"""
        return self.model


if __name__ == "__main__":
    trainer = ModelTrainer()
    trained_model = trainer.train_model()
