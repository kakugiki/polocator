from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_data_generators():
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_directory(
        "data/processed/train",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_generator = val_datagen.flow_from_directory(
        "data/processed/val",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
    )

    test_generator = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_generator.flow_from_directory(
        "data/processed/test",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator
