import unittest
from scripts.image_converter import ImageConverter


class TestImageConverter(unittest.TestCase):
    @unittest.skip("Images converted, not a test")
    def test_convert(self):
        converter = ImageConverter()
        converter.convert_and_move(
            "./data/raw/dog_poop", "./data/processed/dog_poop", "DP"
        )
