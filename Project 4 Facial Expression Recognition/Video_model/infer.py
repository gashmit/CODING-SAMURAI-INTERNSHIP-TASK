import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import argparse

VOXSENSE_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

def preprocess_image(image, target_size=(48, 48)):
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_image_emotion(image_path, model_path="outputs/video_emotion_model.h5"):
    model = load_model(model_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = preprocess_image(image_rgb)
    preds = model.predict(processed)[0]
    label_idx = np.argmax(preds)
    print(f"✅ Predicted emotion: {VOXSENSE_LABELS[label_idx]} ({preds[label_idx]*100:.2f}%)")

def predict_video_emotion(video_path, model_path="outputs/video_emotion_model.h5"):
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = preprocess_image(frame_rgb)
        preds = model.predict(processed)[0]
        label_idx = np.argmax(preds)
        label = VOXSENSE_LABELS[label_idx]
        confidence = preds[label_idx]
        print(f"Frame {frame_count}: {label} ({confidence*100:.2f}%)")
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Emotion Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🎬 Facial Emotion Inference")
    choice = input("Select mode [1 = Image, 2 = Video]: ").strip()

    if choice == "1":
        path = input("Enter path to image: ").strip()
        if os.path.exists(path):
            predict_image_emotion(path)
        else:
            print("❌ File does not exist.")
    elif choice == "2":
        path = input("Enter path to video: ").strip()
        if os.path.exists(path):
            predict_video_emotion(path)
        else:
            print("❌ File does not exist.")
    else:
        print("❌ Invalid selection. Please enter 1 or 2.")
