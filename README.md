## Overview for task 3-4

This folder implements Task 3 and Task 4:

**Task 3** — A MobileNetV2 transfer learning model trained to classify 
fruit and vegetables as fresh or rotten across multiple produce types. 
The model is trained in two phases (frozen base → fine-tuned) and 
evaluated with accuracy, precision, recall, F1, and a confusion matrix.

**Task 4 (XAI)** — model_service.py implements Grad-CAM (Gradient-weighted 
Class Activation Mapping) to explain every prediction. It highlights which 
regions of the image (discolouration, bruising, mould) drove the fresh/rotten 
decision, and generates a plain-English explanation suitable for producers 
and marketplace operators. This satisfies the Fairness, Accountability and 
Trust (FAT) requirements.

### Key files
- train.py — standalone training pipeline (run once to produce the model)
- model_service.py — classification + Grad-CAM XAI (integration point for DESD)
- freshness_model.h5 — pre-trained model (no retraining needed)
- class_names.json — class list matching the model output layer
- results/ — confusion matrix and training history plots



# Task 3 — Freshness Classifier (Setup & Usage)

## Prerequisites

TensorFlow does not support Python 3.12+. You must have **Python 3.11** installed.

Download Python 3.11 from: https://www.python.org/downloads/

---

## Setup

### 1. Navigate to the Task 3 folder

```bash
cd "Task 3"
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

Examples 
```bash
# Fresh examples
venv\Scripts\python "Task 3/model_service.py" "Task 3/dataset/Fruit And Vegetable Diseases Dataset/Apple__Healthy/FreshApple (1).jpg"
venv\Scripts\python "Task 3/model_service.py" "Task 3/dataset/Fruit And Vegetable Diseases Dataset/Banana__Healthy/FreshBanana (1).jpg"
venv\Scripts\python "Task 3/model_service.py" "Task 3/dataset/Fruit And Vegetable Diseases Dataset/Tomato__Healthy/FreshTomato (1).jpg"

# Rotten examples
venv\Scripts\python "Task 3/model_service.py" "Task 3/dataset/Fruit And Vegetable Diseases Dataset/Apple__Rotten/RottenApple (1).jpg"
venv\Scripts\python "Task 3/model_service.py" "Task 3/dataset/Fruit And Vegetable Diseases Dataset/Banana__Rotten/RottenBanana (1).jpg"
venv\Scripts\python "Task 3/model_service.py" "Task 3/dataset/Fruit And Vegetable Diseases Dataset/Tomato__Rotten/RottenTomato (1).jpg"
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
