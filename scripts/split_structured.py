import os
import shutil


def reorganize_dataset(source_dir):
    # Define prefix mapping
    prefix_to_class = {"C_": "control", "DP_": "positive"}

    for split in ["train", "val", "test"]:
        source_path = os.path.join(source_dir, split)

        # Skip if source path doesn't exist
        if not os.path.exists(source_path):
            continue

        # Create subfolders for each class
        for class_name in prefix_to_class.values():
            os.makedirs(os.path.join(source_path, class_name), exist_ok=True)

        # Move files based on their prefixes
        for file in os.listdir(source_path):
            # Only process files in the root of split directory
            if os.path.dirname(os.path.join(source_path, file)) != source_path:
                continue

            # Check for each prefix
            for prefix, class_name in prefix_to_class.items():
                if file.startswith(prefix):
                    # Move the file to appropriate subfolder
                    shutil.move(
                        os.path.join(source_path, file),
                        os.path.join(source_path, class_name, file),
                    )
                    break


# Usage
source_directory = "data/processed"
reorganize_dataset(source_directory)
