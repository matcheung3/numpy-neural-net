# Neural Network from Scratch with NumPy

This repository contains a NumPy-only implementation of a basic feedforward neural network. The network supports training via mini-batch gradient descent and is capable of multiclass classification using softmax activation and cross-entropy loss.

> ⚠️ **Note**: This implementation is designed for datasets with **3 numerical input features** and **class labels (0, 1, 2)** as the last column.

---

## 🧠 Architecture

The neural network is defined layer-wise as follows:

```python
nn_architecture = [
    {"input_dim": 3, "output_dim": 4},
    {"input_dim": 4, "output_dim": 8},
    {"input_dim": 8, "output_dim": 3}
]
```

It includes two hidden layers using ReLU activation and a softmax output layer for 3-class classification.

---

## 📊 Features

- Forward and backward propagation
- ReLU and softmax activation functions
- Cross-entropy loss
- Batch-wise training (mini-batch gradient descent)
- Accuracy evaluation on training and test sets
- Manual weight and bias updates (no frameworks)

---

## 🗂️ Project Structure

```
.
├── nn_from_scratch.py   # Main code implementing the neural network
└── README.md            # This file
```

---

## 📁 Data Format

The expected dataset format:
- CSV file (e.g., `xyz.csv`).
- Each row: 3 numerical features + 1 integer class label (0, 1, or 2).
- Data is expected to contain 800 rows (first 400 used for training, remaining 400 for testing).

---

## ▶️ How to Run

Make sure you have NumPy installed:

```bash
pip install numpy
```

Then run the script:

```bash
python nn_from_scratch.py
```

Output will show:
- Loss per epoch
- Training accuracy per epoch
- Final test set accuracy

---

## 📚 Academic Context

This project was completed as part of a course assignment focused on understanding the inner workings of neural networks without using any external machine learning libraries.

