from PIL import Image
import os

path = "path/to/your/dataset"
for filename in os.listdir(path):
    if filename.endswith(".heic"):
        img = Image.open(os.path.join(path, filename))
        new_filename = os.path.splitext(filename)[0] + ".jpg"  # or '.png'
        img.save(os.path.join(path, new_filename))


path = "path/to/your/dataset"
for i, filename in enumerate(os.listdir(path)):
    if filename.startswith("2024"):
        category = "DP" if "dog" in filename.lower() else "C"
        new_filename = f"{category}_{i+1:03}.jpg"  # Assuming all files are jpg
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
