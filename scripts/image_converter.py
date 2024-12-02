import os
from PIL import Image
from pillow_heif import register_heif_opener


class ImageConverter:
    def __init__(self):
        pass

    def convert_and_move(self, source_dir, target_dir, category):
        # Register HEIF opener to handle HEIC files
        register_heif_opener()

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for filename in os.listdir(source_dir):
            if filename.endswith(".heic"):
                filepath = os.path.join(source_dir, filename)
                abspath = os.path.abspath(filepath)  # Convert to absolute path
                if os.path.exists(abspath):
                    try:
                        img = Image.open(abspath)
                        new_filename = (
                            f"{category}_{len(os.listdir(target_dir)) + 1}.jpg"
                        )
                        img.save(os.path.join(target_dir, new_filename))
                        print(f"Converted and moved: {filename} -> {new_filename}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                else:
                    print(f"File not found: {abspath}")


# Example usage:
# converter = ImageConverter()
# converter.convert_and_move("./data/raw/dog_poop", "./data/processed/dog_poop", "DP")
