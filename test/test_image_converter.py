import unittest
from scripts.image_converter import ImageConverter


class TestImageConverter(unittest.TestCase):
    @unittest.skip("Images converted, not a test")
    def test_convert(self):
        converter = ImageConverter()
        converter.convert_and_move(
            "./data/raw/dog_poop", "./data/processed/dog_poop", "DP"
        )
        
    def test_class(self):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator()
        train_generator = datagen.flow_from_directory(
            'data/processed/train',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )

        # This will show the mapping between class indices and class names
        print(train_generator.class_indices)
        
        self.assertEqual(
            [k for k, v in train_generator.class_indices.items() if v == 0][0], 
            'control'
        )
