import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# VoxSense label order
VOXSENSE_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def evaluate_video_emotion_model(model_path="outputs/video_emotion_model.h5", data_path="processed_folder_dataset"):
    """
    Load the trained model and evaluate it on the test set.
    """
    model = load_model(model_path)
    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))

    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Report
    print("\nClassification Report:\n")
    from sklearn.utils.multiclass import unique_labels
    actual_labels = unique_labels(y_true, y_pred)
    label_names = [VOXSENSE_LABELS[i] for i in actual_labels]
    report = classification_report(y_true, y_pred, labels=actual_labels, target_names=label_names)
    print(report)
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_report.txt", "w") as f:
        f.write("Classification Report:\n\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=VOXSENSE_LABELS,
                yticklabels=VOXSENSE_LABELS)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

    print("✅ Evaluation complete. Saved confusion matrix as 'outputs/confusion_matrix.png'.")

if __name__ == "__main__":
    evaluate_video_emotion_model()