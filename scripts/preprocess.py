import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Random seed for reproducibility
random.seed(42)

# Dataset paths
RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

# train/val/test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def make_dirs():
    """
    Create output directories for train/val/test splits. Each split has two subfolders: '0' (benign) and '1' (malignant).
    """
    for split in ["train", "val", "test"]:
        for cls in ["0", "1"]:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def split_patients():
    """
    Split patient IDs into train/val/test sets. This ensures that patients are split, not individual images, to prevent data leakage between sets.
    """
    patients = [p for p in RAW_DATA_DIR.iterdir() if p.is_dir()] # Get all patient directories
    patient_ids = [p.name for p in patients]

    train_ids, temp_ids = train_test_split(patient_ids, train_size=train_ratio, random_state=42) # Split into train and (val+test)
    val_ids, test_ids = train_test_split(temp_ids, test_size=test_ratio/(val_ratio+test_ratio), random_state=42) # Split (val+test) into val and test

    return set(train_ids), set(val_ids), set(test_ids)

def copy_images(patient_ids, split_name):
    """
    Copy images for a split (train/val/test) into the data/processed directory
    """
    for pid in patient_ids:
        patient_folder = RAW_DATA_DIR / pid
        if not patient_folder.exists():
            continue

        for cls in ["0", "1"]: # Loop over 0 and 1 classes
            class_folder = patient_folder / cls
            if not class_folder.exists():
                continue

            for img_file in class_folder.glob("*.png"): # Copy all PNG images into the correct folder
                dest = OUTPUT_DIR / split_name / cls / img_file.name
                shutil.copy(img_file, dest)

def main():
    """
    Creates directory structure
    Splits patients into train/val/test
    Copies images into respective split folders
    """
    make_dirs()
    train_ids, val_ids, test_ids = split_patients()

    print(f"Train patients: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    copy_images(train_ids, "train")
    copy_images(val_ids, "val")
    copy_images(test_ids, "test")

    print("Dataset preprocessing complete!")

if __name__ == "__main__":
    main()