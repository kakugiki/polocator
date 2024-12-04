import os
import random
import shutil

def consolidate_and_split_dataset(control_dir, positive_dir, train_dir, val_dir, test_dir, split_ratio=[0.7, 0.15, 0.15]):
    # List all files in both directories
    control_files = [os.path.join(control_dir, f) for f in os.listdir(control_dir) if os.path.isfile(os.path.join(control_dir, f))]
    positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if os.path.isfile(os.path.join(positive_dir, f))]
    
    # Combine into one list with class information
    all_files = [(f, 'control') for f in control_files] + [(f, 'positive') for f in positive_files]
    
    # Shuffle the files
    random.shuffle(all_files)
    
    # Calculate split points
    total_files = len(all_files)
    train_split = int(total_files * split_ratio[0])
    val_split = train_split + int(total_files * split_ratio[1])
    
    # Split the data
    train_files = all_files[:train_split]
    val_files = all_files[train_split:val_split]
    test_files = all_files[val_split:]
    
    # Ensure output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Move files to respective directories
    for file, class_name in train_files:
        shutil.copy(file, train_dir)
    for file, class_name in val_files:
        shutil.copy(file, val_dir)
    for file, class_name in test_files:
        shutil.copy(file, test_dir)

# Usage
control_directory = 'data/processed/control'
positive_directory = 'data/processed/dog_poop'
train_directory = 'data/processed/train'
val_directory = 'data/processed/val'
test_directory = 'data/processed/test'

consolidate_and_split_dataset(control_directory, positive_directory, train_directory, val_directory, test_directory)
