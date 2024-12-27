from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


def get_data_generators():
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and test data generators (no augmentation, only preprocessing)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        "data/processed/train",
        target_size=(100, 100),
        batch_size=32,
        class_mode="categorical",
    )

    val_generator = val_datagen.flow_from_directory(
        "data/processed/val",
        target_size=(100, 100),
        batch_size=32,
        class_mode="categorical",
    )

    test_generator = test_datagen.flow_from_directory(
        "data/processed/test",
        target_size=(100, 100),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator
