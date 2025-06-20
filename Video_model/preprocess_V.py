import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# VoxSense labels (must match your folder names exactly or map accordingly)
VOXSENSE_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
VOXSENSE_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(VOXSENSE_LABELS)}

def load_images_from_folder(root_dir: str, img_size=(48,48)) -> (np.ndarray, np.ndarray):
    images = []
    labels = []
    for label_name in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        # Map folder label to VoxSense label index
        if label_name not in VOXSENSE_LABEL_TO_INDEX:
            print(f"Warning: Label '{label_name}' not in VoxSense label set, skipping.")
            continue
        label_idx = VOXSENSE_LABEL_TO_INDEX[label_name]

        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            try:
                img = Image.open(file_path).convert('RGB')  # RGB
                img = img.resize(img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(label_idx)
            except Exception as e:
                print(f"Failed to process image {file_path}: {e}")

    X = np.array(images)
    y = to_categorical(labels, num_classes=len(VOXSENSE_LABELS))
    return X, y

def preprocess_dataset(root_path: str = "video_datasets", output_dir: str = "processed_folder_dataset") -> None:
    """
    Load and preprocess images from folder-structured dataset.

    Args:
        root_path (str): Root folder containing 'train' and 'test' folders.
        output_dir (str): Directory to save processed numpy arrays.
    """
    train_path = os.path.join(root_path, "train")
    test_path = os.path.join(root_path, "test")

    print("Loading training data...")
    X_train, y_train = load_images_from_folder(train_path)
    print(f"Training data: {X_train.shape}, {y_train.shape}")

    print("Loading test data...")
    X_test, y_test = load_images_from_folder(test_path)
    print(f"Test data: {X_test.shape}, {y_test.shape}")

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print(f"✅ Dataset processed and saved in '{output_dir}'.")


# --- RAF-DB loader and combined preprocessor ---
def load_rafdb_images_from_folder(root_dir: str="/Users/ashmitgupta/Facial Recogni/video_datasets/RAF-DB Dataset", img_size=(48, 48)) -> (np.ndarray, np.ndarray):
    images = []
    labels = []
    for label_name in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_name)
        if not os.path.isdir(label_path):
            continue
        if label_name not in VOXSENSE_LABEL_TO_INDEX:
            print(f"Warning: Label '{label_name}' not in VoxSense label set, skipping.")
            continue
        label_idx = VOXSENSE_LABEL_TO_INDEX[label_name]
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(label_idx)
            except Exception as e:
                print(f"Failed to process RAF-DB image {file_path}: {e}")
    X = np.array(images)
    y = to_categorical(labels, num_classes=len(VOXSENSE_LABELS))
    return X, y

def preprocess_dataset(root_path: str = "video_datasets", rafdb_path: str = "/Users/ashmitgupta/Facial Recogni/video_datasets/RAF-DB Dataset", output_dir: str = "processed_folder_dataset") -> None:
    train_path = os.path.join(root_path, "train")
    test_path = os.path.join(root_path, "test")

    print("Loading training data from main dataset...")
    X_train_main, y_train_main = load_images_from_folder(train_path)
    print(f"Main training data: {X_train_main.shape}, {y_train_main.shape}")

    print("Loading test data from main dataset...")
    X_test_main, y_test_main = load_images_from_folder(test_path)
    print(f"Main test data: {X_test_main.shape}, {y_test_main.shape}")

    # Load additional RAF-DB data if available
    raf_train_path = os.path.join(rafdb_path, "train")
    raf_test_path = os.path.join(rafdb_path, "test")

    X_train_raf, y_train_raf = load_rafdb_images_from_folder(raf_train_path)
    X_test_raf, y_test_raf = load_rafdb_images_from_folder(raf_test_path)

    print(f"RAF-DB training data: {X_train_raf.shape}, {y_train_raf.shape}")
    print(f"RAF-DB test data: {X_test_raf.shape}, {y_test_raf.shape}")

    # Load CK+ dataset
    ckplus_path = os.path.join("video_datasets", "CK+")
    X_ckplus, y_ckplus = load_rafdb_images_from_folder(ckplus_path)
    print(f"CK+ data: {X_ckplus.shape}, {y_ckplus.shape}")
    if X_ckplus.ndim == 1:
        X_ckplus = np.empty((0, 48, 48, 3), dtype=np.float32)

    # Merge datasets
    X_train = np.concatenate([X_train_main, X_train_raf, X_ckplus], axis=0)
    y_train = np.concatenate([y_train_main, y_train_raf, y_ckplus], axis=0)
    X_test = np.concatenate([X_test_main, X_test_raf], axis=0)
    y_test = np.concatenate([y_test_main, y_test_raf], axis=0)

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    print(f"✅ Combined dataset (main + RAF-DB + CK+) saved in '{output_dir}'.")

if __name__ == "__main__":
    preprocess_dataset("video_datasets", "/Users/ashmitgupta/Facial Recogni/video_datasets/RAF-DB Dataset")