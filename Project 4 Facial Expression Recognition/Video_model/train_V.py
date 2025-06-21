import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime
from model_V import build_video_emotion_model

def train_video_emotion_model(
    data_path="processed_folder_dataset",
    model_path="outputs/video_emotion_model.h5",
    batch_size=32,
    epochs=50
):
    """
    Train the facial emotion recognition CNN using preprocessed data.

    Args:
        data_path (str): Path to the folder with X_train.npy and y_train.npy files.
        model_path (str): Path to save the trained model.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
    """
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # Load the data
    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    print(f"Label distribution (train): {np.sum(y_train, axis=0)}")
    print(f"Label distribution (test): {np.sum(y_test, axis=0)}")

    # Build the model
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]
    model = build_video_emotion_model(input_shape=input_shape, num_classes=num_classes)

    model.summary()

    log_dir = f"outputs/logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )

    print(f"✅ Model trained and saved to {model_path}")

    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("outputs/training_plot.png")
    plt.show()

if __name__ == "__main__":
    train_video_emotion_model()