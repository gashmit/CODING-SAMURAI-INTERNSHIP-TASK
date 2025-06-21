

import os
import shutil

# Map CK+ folder names to VoxSense labels
CKPLUS_TO_VOXSENSE = {
    'anger': 'angry',
    'happy': 'happy',
    'sadness': 'sad',
    'disgust': 'disgust',
    'fear': 'fear',
    'surprise': 'surprise',
    'contempt': None  # ignored
}

VOXSENSE_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def fix_ckplus_labels(root_dir: str):
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        mapped = CKPLUS_TO_VOXSENSE.get(folder.lower())
        if mapped is None:
            print(f"❌ Removing/ignoring: {folder}")
            shutil.rmtree(folder_path)
            continue

        if mapped != folder:
            new_path = os.path.join(root_dir, mapped)
            if os.path.exists(new_path):
                for f in os.listdir(folder_path):
                    shutil.move(os.path.join(folder_path, f), os.path.join(new_path, f))
                os.rmdir(folder_path)
                print(f"🔁 Merged '{folder}' → '{mapped}'")
            else:
                shutil.move(folder_path, new_path)
                print(f"🔁 Renamed '{folder}' → '{mapped}'")
        else:
            print(f"✅ Already standardized: {folder}")

    # Add empty folders for missing labels
    for label in VOXSENSE_LABELS:
        target = os.path.join(root_dir, label)
        os.makedirs(target, exist_ok=True)

    print("\n✅ CK+ labels standardized to VoxSense format.")

if __name__ == "__main__":
    fix_ckplus_labels("video_datasets/CK+")