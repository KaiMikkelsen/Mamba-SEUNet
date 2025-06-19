import os
import shutil
import json
from sklearn.model_selection import train_test_split
import argparse

# --- Configuration ---
# Your existing generated audio data directories
CLEAN_SOURCE_DIR = "/Users/kaimikkelsen/guitar_noise/data/generated_datasets/training_clean_targets"
NOISY_SOURCE_DIR = "/Users/kaimikkelsen/guitar_noise/data/generated_datasets/training_noisy_inputs" # Assuming this is your noisy inputs

# New root directory for your split dataset
OUTPUT_DATASET_ROOT = "mamba_guitar_dataset"

# Define the target split directories relative to OUTPUT_DATASET_ROOT
SPLIT_DIRS = {
    "train_clean": os.path.join(OUTPUT_DATASET_ROOT, "clean_train"),
    "train_noisy": os.path.join(OUTPUT_DATASET_ROOT, "noisy_train"),
    "valid_clean": os.path.join(OUTPUT_DATASET_ROOT, "clean_valid"),
    "valid_noisy": os.path.join(OUTPUT_DATASET_ROOT, "noisy_valid"),
    "test_clean": os.path.join(OUTPUT_DATASET_ROOT, "clean_test"),
    "test_noisy": os.path.join(OUTPUT_DATASET_ROOT, "noisy_test"),
}

# --- Functions (adapted from your provided code and standard practices) ---

def list_wav_filenames_in_directory(directory_path):
    """Lists only .wav filenames (basename) in a directory."""
    filenames = set() # Use a set to store unique basenames
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return []
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith('.wav'):
                filenames.add(filename) # Add just the filename, not the full path
    return sorted(list(filenames)) # Return a sorted list of unique basenames

def save_paths_to_json(paths_list, output_file):
    """Saves a list of file paths to a JSON file."""
    with open(output_file, 'w') as json_file:
        json.dump(paths_list, json_file, indent=4)
    print(f"Generated JSON: {output_file} with {len(paths_list)} entries.")

def create_and_move_splits():
    """
    Orchestrates the creation of split directories, moving files,
    and generating JSON manifests.
    """
    # 1. Get all unique filenames (identifiers) from the clean source directory
    # We use clean as the primary source of filenames, assuming noisy has exact matches.
    all_filenames = list_wav_filenames_in_directory(CLEAN_SOURCE_DIR)

    if not all_filenames:
        print(f"No .wav files found in {CLEAN_SOURCE_DIR}. Please ensure your generated audio is there.")
        return

    print(f"Found {len(all_filenames)} unique audio file identifiers to split.")

    # 2. Create all destination directories
    for path in SPLIT_DIRS.values():
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    # 3. Perform the train/validation/test split on filenames
    # Using 70/15/15 split (train/val/test)
    train_filenames, test_filenames = train_test_split(all_filenames, test_size=0.15, random_state=42)
    train_filenames, valid_filenames = train_test_split(train_filenames, test_size=(0.15 / (1 - 0.15)), random_state=42)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_filenames)} files")
    print(f"  Validation: {len(valid_filenames)} files")
    print(f"  Test: {len(test_filenames)} files")

    # Dictionary to store absolute paths for JSON generation
    json_data = {
        "train_clean": [], "train_noisy": [],
        "valid_clean": [], "valid_noisy": [],
        "test_clean": [], "test_noisy": [],
    }

    # 4. Move files and collect new absolute paths
    print("\nMoving files and collecting paths...")

    # Train Split
    for filename in train_filenames:
        # Source paths
        src_clean = os.path.join(CLEAN_SOURCE_DIR, filename)
        src_noisy = os.path.join(NOISY_SOURCE_DIR, filename)
        
        # Destination paths
        dest_clean = os.path.join(SPLIT_DIRS["train_clean"], filename)
        dest_noisy = os.path.join(SPLIT_DIRS["train_noisy"], filename)

        # Move files
        shutil.move(src_clean, dest_clean)
        shutil.move(src_noisy, dest_noisy)

        # Add absolute paths to JSON data lists
        json_data["train_clean"].append(os.path.abspath(dest_clean))
        json_data["train_noisy"].append(os.path.abspath(dest_noisy))
    print(f"  Moved {len(train_filenames)} train pairs.")

    # Validation Split
    for filename in valid_filenames:
        src_clean = os.path.join(CLEAN_SOURCE_DIR, filename)
        src_noisy = os.path.join(NOISY_SOURCE_DIR, filename)

        dest_clean = os.path.join(SPLIT_DIRS["valid_clean"], filename)
        dest_noisy = os.path.join(SPLIT_DIRS["valid_noisy"], filename)

        shutil.move(src_clean, dest_clean)
        shutil.move(src_noisy, dest_noisy)

        json_data["valid_clean"].append(os.path.abspath(dest_clean))
        json_data["valid_noisy"].append(os.path.abspath(dest_noisy))
    print(f"  Moved {len(valid_filenames)} validation pairs.")

    # Test Split
    for filename in test_filenames:
        src_clean = os.path.join(CLEAN_SOURCE_DIR, filename)
        src_noisy = os.path.join(NOISY_SOURCE_DIR, filename)

        dest_clean = os.path.join(SPLIT_DIRS["test_clean"], filename)
        dest_noisy = os.path.join(SPLIT_DIRS["test_noisy"], filename)

        shutil.move(src_clean, dest_clean)
        shutil.move(src_noisy, dest_noisy)

        json_data["test_clean"].append(os.path.abspath(dest_clean))
        json_data["test_noisy"].append(os.path.abspath(dest_noisy))
    print(f"  Moved {len(test_filenames)} test pairs.")

    # 5. Generate JSON files
    print("\nGenerating JSON manifest files...")
    save_paths_to_json(json_data["train_clean"], os.path.join(OUTPUT_DATASET_ROOT, "train_clean.json"))
    save_paths_to_json(json_data["train_noisy"], os.path.join(OUTPUT_DATASET_ROOT, "train_noisy.json"))
    save_paths_to_json(json_data["valid_clean"], os.path.join(OUTPUT_DATASET_ROOT, "valid_clean.json"))
    save_paths_to_json(json_data["valid_noisy"], os.path.join(OUTPUT_DATASET_ROOT, "valid_noisy.json"))
    save_paths_to_json(json_data["test_clean"], os.path.join(OUTPUT_DATASET_ROOT, "test_clean.json"))
    save_paths_to_json(json_data["test_noisy"], os.path.join(OUTPUT_DATASET_ROOT, "test_noisy.json"))

    print("\nDataset splitting and JSON generation complete!")
    print(f"All processed files and JSONs are located in: {os.path.abspath(OUTPUT_DATASET_ROOT)}")
    print("\nIMPORTANT: Remember to adjust your Mamba-SEUNet configuration (e.g., `prepath` or direct JSON paths) to point to this new `mamba_guitar_dataset` directory.")

if __name__ == '__main__':
    # You can optionally use argparse here if you want to make source/dest paths configurable
    # For now, it uses the hardcoded paths at the top of the script.
    # If you intend to run this as a standalone script for your project,
    # you might add:
    # parser = argparse.ArgumentParser(description="Split audio dataset and generate JSONs.")
    # parser.add_argument('--clean_src', type=str, default=CLEAN_SOURCE_DIR, help='Path to source clean audio directory.')
    # parser.add_argument('--noisy_src', type=str, default=NOISY_SOURCE_DIR, help='Path to source noisy audio directory.')
    # parser.add_argument('--output_root', type=str, default=OUTPUT_DATASET_ROOT, help='Root directory for output splits and JSONs.')
    # args = parser.parse_args()
    # CLEAN_SOURCE_DIR = args.clean_src
    # NOISY_SOURCE_DIR = args.noisy_src
    # OUTPUT_DATASET_ROOT = args.output_root # Re-assign for consistency if using args

    create_and_move_splits()