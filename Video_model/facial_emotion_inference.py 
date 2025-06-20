

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# VoxSense emotion labels
VOXSENSE_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def load_facial_emotion_model(model_path: str):
    """Load the trained facial emotion model."""
    model = load_model(model_path)
    return model

def preprocess_image(img_path: str, target_size=(48, 48)) -> np.ndarray:
    """Load and preprocess image for model inference."""
    img = Image.open(img_path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (48,48,1)
    img_array = np.expand_dims(img_array, axis=0)   # (1,48,48,1)
    return img_array

def predict_emotion(model, img_array: np.ndarray) -> tuple[str, float]:
    """Predict emotion from preprocessed image array."""
    preds = model.predict(img_array)

    print("Emotion Probabilities:")
    for i, prob in enumerate(preds[0]):
        print(f"{VOXSENSE_LABELS[i]}: {prob * 100:.2f}%")

    confidence = float(np.max(preds)) * 100
    emotion = VOXSENSE_LABELS[int(np.argmax(preds))]
    return emotion, confidence

if __name__ == "__main__":
    model_path = "video_emotion_model.h5"  # Update this path if needed
    test_image_path = "/Users/ashmitgupta/voxsense.1/train_00009_aligned.jpg"      # Update this with your test image

    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Image not found: {test_image_path}")

    model = load_facial_emotion_model(model_path)
    img_array = preprocess_image(test_image_path)
    emotion, confidence = predict_emotion(model, img_array)

    print(f"Predicted Emotion: {emotion} ({confidence:.2f}%)")
