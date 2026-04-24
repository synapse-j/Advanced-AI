


# Task 3 — Freshness Classifier (Setup & Usage)

## Prerequisites

TensorFlow does not support Python 3.12+. You must have **Python 3.11** installed.

Download Python 3.11 from: https://www.python.org/downloads/

---

## Setup

### 1. Navigate to the Task 3 folder

```bash
cd "C:\Users\Vinnie\Documents\Uni\AI_2\Advanced-AI\Task 3"
```

### 2. Create a virtual environment using Python 3.11

```bash
py -3.11 -m venv venv
```

### 3. Install required libraries

```bash
venv\Scripts\pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
```

---

## Running the programs

> **Important:** Always use `venv\Scripts\python` instead of `python` to ensure the correct Python version and installed packages are used.

### Classify an image (model_service.py)

The trained model (`freshness_model.h5`) is already saved. You do not need to retrain.

```bash
venv\Scripts\python model_service.py "dataset/Fruit And Vegetable Diseases Dataset/Apple__Healthy/FreshApple (1).jpg"
```

Replace the image path with any image from the dataset folder.

### Retrain the model (train.py)

Only needed if you want to retrain from scratch.

```bash
venv\Scripts\python train.py
```

---

## Required Libraries

| Library | Purpose |
|---|---|
| tensorflow | Model training and inference |
| opencv-python | Grad-CAM heatmap generation |
| numpy | Array operations |
| pandas | Dataset loading and manipulation |
| matplotlib | Training history plots |
| seaborn | Confusion matrix heatmap |
| scikit-learn | Train/test splits and metrics |

---

## Dataset

The dataset should be placed at:

```
Task 3/dataset/Fruit And Vegetable Diseases Dataset/<ClassName>/
```

Download from: https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten
