from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.layers import Flatten


def create_model():
    # Load ResNet50 with pre-trained weights, excluding top layers
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=(100, 100, 3)
    )

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom top layers for our classification task
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile with rmsprop optimizer as used in training
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
