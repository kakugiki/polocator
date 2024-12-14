from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
from PIL import Image
import pillow_heif
import os


def load_image(image_path, target_size):
    """
    Load and convert image to RGB format, handling both HEIC and regular formats

    Args:
        image_path (str): Path to the image
        target_size (tuple): Desired output size (width, height)

    Returns:
        PIL.Image: Converted and resized image
    """
    if image_path.lower().endswith('.heic'):
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
    else:
        image = Image.open(image_path)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize image
    image = image.resize(target_size)
    return image


model = load_model("models/best_model.keras")

folder_path = "data/processed/test/control"
results = []
filenames = []
target_size = (224, 224)

for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg", ".heic")):  # Adjust file types as needed
        img_path = os.path.join(folder_path, filename)

        # Load and preprocess image
        img = load_image(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize if you did this during training

        # Predict
        prediction = model.predict(img_array)

        # Assuming classification task, get the predicted class
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_probabilities = prediction[0]  # Probabilities for each class

        # Store results
        results.append(
            class_probabilities
        )  # or [predicted_class] if you just want the class
        filenames.append(filename)

# Create DataFrame
if not results:
    print("No results found. Check if any valid images were processed.")
    exit(1)

print(f"Number of files processed: {len(filenames)}")
print(f"Shape of first result: {results[0].shape if results else 'No results'}")

df = pd.DataFrame(
    results, index=filenames, columns=[f"Class_{i}" for i in range(len(results[0]))]
)
df["Predicted_Class"] = df.apply(
    lambda row: np.argmax(row), axis=1
)  # Add column for predicted class

csv_path = "models/prediction_results.csv"
df.to_csv(csv_path)
