# 💊 Smart Video Recognition System: Pill Intake & Face Recognition with TensorFlow

This project combines **video-based action recognition**, **face detection and recognition**, and **real-time inference** using TensorFlow and OpenCV.  
It includes training pipelines, live detection scripts, face embedding generators, and conversion tools to TensorFlow Lite.

---

## 📦 Project Structure

| File/Folder                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `main.py`                   | Main training script for video action classifier using TensorFlow           |
| `frame_generator.py`        | Generator that extracts video frames and associates them with labels        |
| `Detector.py`               | Real-time pill-taking action detector using saved TensorFlow model          |
| `Pill_Swallow_Trainer.py`   | Model trainer using MobileNetV2 + custom classification layers              |
| `Pill_Swallow_Recognizer.py`| Lightweight live video recognizer with OpenCV and Keras model               |
| `Face_detectorOs.py`        | Face embedding and recognition using `facenet-pytorch` and webcam input     |
| `Face_Trainer.py`           | Trainer script to build `face-trainner.yml` using OpenCV's LBPH algorithm   |
| `Face_RecogPi.py`           | Lightweight face recognition using OpenCV and face-trainner model           |
| `converter.py`              | Converts a trained TensorFlow model into TensorFlow Lite format             |
| `categorize.py`             | Batch rename utility for video datasets                                     |
| `utils.py` (assumed)        | Helper methods for loading video frames (used in `frame_generator.py`)      |

---

## 🧠 Features

- ✅ **Action Recognition**: Detects pill intake actions from short video sequences
- ✅ **Face Recognition**: Identifies individuals using pre-trained LBPH or Facenet embeddings
- ✅ **TensorFlow Model Training**: Supports video-based classifier and MobileNetV2 fine-tuning
- ✅ **Live Inference**: Real-time prediction through webcam
- ✅ **Model Export**: Convert trained models to `.tflite` for deployment

---

## 🧪 Datasets

- `videos/train/`, `videos/val/`: Organize videos into class folders (e.g. `positive/`, `negative/`)
- `photos/`: Face images per user (used for LBPH and Facenet recognition)
- `face-trainner.yml`: Saved OpenCV face recognizer model
- Live webcam is used for inference during evaluation.

---

## 🛠️ Requirements

```bash
pip install tensorflow opencv-python facenet-pytorch numpy pandas pillow
