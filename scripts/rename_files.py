import os
import re

def rename_files(directory, pattern, prefix):
    """Renames files in a directory matching a pattern.

    Args:
        directory: The path to the directory.
        pattern: The regular expression pattern to match filenames.
        prefix: The prefix to use for the new filenames.
    """
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            number = int(match.group(1))
            new_filename = f"{prefix}_{number:03d}.jpg"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            try:
                os.rename(old_path, new_path)
                print(f"Renamed '{filename}' to '{new_filename}'")
            except OSError as e:
                print(f"Error renaming '{filename}': {e}")


if __name__ == "__main__":
    directory = "data/processed/dog_poop"
    pattern = r"C_(\d+)\.jpg"
    prefix = "DP"
    rename_files(directory, pattern, prefix)
