# 🚗 CARLA Road Segmentation with U-Net

This project implements a semantic segmentation pipeline using a custom-built **U-Net** architecture, trained on synthetic driving scenes from the **CARLA simulator**. The model learns to distinguish roads, sidewalks, and other driveable areas - pixel by pixel. A foundational step toward autonomous driving systems.

---

## 📁 Repository Structure

```bash
carla-road-segmentation-unet/
│
├── data/ # Dataset: RGB images and segmentation masks from CARLA
│ ├── CameraRGB/ # Input images (simulated car camera views)
│ └── CameraMask/ # Ground truth segmentation masks
│
├── images/ # Visual diagrams
│ ├── encoder.png # Encoder block overview
│ ├── decoder.png # Decoder block overview
│ ├── unet.png # Full U-Net architecture
│ └── carseg.png # Sample predictions
│
├── saved_model/
│ └── unet_model.h5 # Trained U-Net model
│
├── notebook.ipynb # Complete pipeline (training, evaluation, prediction)
└── README.md
```

---

## 🧠 What’s Inside?

- **Data Preprocessing**: All CARLA `.png` images are resized, normalized, and loaded with `tf.data`.
- **U-Net Architecture**: Built from scratch with TensorFlow/Keras. Includes skip connections, max-pooling, dropout, and transposed convolutions.
- **Training**: Uses `SparseCategoricalCrossentropy` loss and tracks accuracy.
- **Evaluation**: Visualizes prediction vs ground truth for side-by-side comparison.
- **Model Saving**: Trained model is saved in `.h5` format for easy reuse or fine-tuning.

---

## 📷 Example Results

Below is a sample from the CARLA dataset:

| Input Image | Ground Truth | Model Prediction |
|-------------|--------------|------------------|
| ![img](images/carseg.png) | Semantic classes (gray) | Segmented output (colored) |

---

## 🏗 How U-Net Works (Quick Summary)

- **Encoder**: Repeated Conv2D layers + MaxPooling shrink the image and extract features.
- **Decoder**: Upsamples the feature maps and concatenates with corresponding encoder layers (skip connections).
- **Output**: 1×1 Conv that outputs a class for each pixel in the image.

> U-Net is especially good for tasks where spatial details matter — like road segmentation in self-driving.

---
