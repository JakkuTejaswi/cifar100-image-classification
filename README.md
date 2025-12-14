# CIFAR-100 Image Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-100 dataset** using **TensorFlow and Keras**.

The CIFAR-100 dataset consists of small color images belonging to **100 different classes**, making it a challenging multi-class image classification problem.

---

## 📊 Dataset Information

- **Dataset:** CIFAR-100
- **Source:** `tensorflow.keras.datasets.cifar100`
- **Image size:** 32 × 32 RGB images
- **Number of classes:** 100
- **Training samples:** 50,000
- **Test samples:** 10,000

Each image belongs to one of 100 fine-grained object categories.

---

## 🧠 Model Architecture

The CNN model consists of:

- Input Layer: 32×32×3 images
- Convolutional Layers with ReLU activation
- Batch Normalization for faster convergence
- Max Pooling layers for spatial reduction
- Fully Connected Dense layers
- Dropout for regularization
- Output Layer: 100 neurons with Softmax activation

### Activation Functions Used
- **ReLU** → Hidden layers
- **Softmax** → Output layer (multi-class classification)

---

## ⚙️ Model Compilation

- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Metric:** Accuracy

---

## 📈 Training Performance

- Training and validation accuracy are visualized using **learning curves**
  
Example plot:
- Training Accuracy vs Validation Accuracy
- Training Loss vs Validation Loss

---

## 🚀 How to Run the Project

### 1️. Clone the repository
git clone https://github.com/JakkuTejaswi/cifar100-image-classification.git

### 2. Install dependencies
pip install tensorflow numpy matplotlib

## 3. Run the notebook or script
jupyter notebook CNN.ipynb


