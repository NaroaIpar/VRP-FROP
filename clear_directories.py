# File: clear_directories.py

import os
import shutil

# Directories to clear
directories = [
    "./routes_imgs/beam",
    "./routes_imgs/greedy",
    "./routes_imgs/random",
    "./checkpoints/training_metrics_imgs/beam",
    "./checkpoints/training_metrics_imgs/greedy",
    "./checkpoints/training_metrics_imgs/random"
]

def clear_directory(path: str):
    if not os.path.exists(path):
        print(f"Directory does not exist: {path}")
        return

    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.unlink(entry_path)
            elif os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            print(f"Succes in deleteing: {entry_path}")
        except Exception as e:
            print(f"Failed to delete {entry_path}: {e}")

def main():
    for directory in directories:
        clear_directory(directory)
    print("Selected directories have been cleared.")

if __name__ == "__main__":
    main()
