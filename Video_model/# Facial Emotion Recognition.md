# Facial Emotion Recognition (Internship Project)

This project performs facial emotion recognition using deep learning. It supports training from scratch using FER2013, RAF-DB, and CK+ datasets, and allows inference from both images and video files.

## 📁 Project Structure

```
facial_emotion_recognition/
├── data/                  # Place your datasets here
├── outputs/               # Trained models, logs, plots
├── src/                   # Training and preprocessing scripts
├── infer.py              # Run emotion prediction on images/videos
├── train_V.py            # Train the model from scratch
├── evaluate.py           # Evaluate the model on test set
└── requirements.txt       # Install dependencies
```

## 🧪 How to Run

### 1. Preprocess Datasets
Ensure your FER2013, RAF-DB, and CK+ datasets are available and modify paths in `data_loader.py` if needed.

### 2. Train the Model
```bash
python train_V.py
```

Trained model will be saved to:
```
outputs/video_emotion_model.h5
```

### 3. Evaluate the Model
```bash
python evaluate.py
```

This generates a confusion matrix in:
```
outputs/confusion_matrix.png
```

### 4. Inference from Image
```bash
python infer.py --image path/to/image.jpg
```

### 5. Inference from Video
```bash
python infer.py --video path/to/video.mp4
```

Press `Q` to exit video window.

## 📊 TensorBoard (Optional)
To view training logs:
```bash
tensorboard --logdir=outputs/logs
```

## ✅ Emotion Labels Used
['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

---
