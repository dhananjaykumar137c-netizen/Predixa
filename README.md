<<<<<<< HEAD
# PREDIXA: Alternative Data Intelligence Pipeline 🚀

Welcome to **PREDIXA**, a machine learning pipeline designed to clean, process, and analyze large e-commerce product datasets, and train a **multi-task PyTorch model** using a shared `distilbert-base-uncased` backbone.

The pipeline performs two prediction tasks simultaneously based on product reviews (concatenating the review body and summary):
1. **Category Classification** (Binary classification): Classifies reviews into **All Beauty** (0) or **Appliances** (1).
2. **Rating Prediction** (Regression): Predicts a star rating from **1.0 to 5.0**.
=======
# Predixa

**Predixa** is a multi-task deep learning project that classifies product reviews and predicts star ratings using a fine-tuned **DistilBERT** model. It demonstrates end-to-end NLP workflows including data cleaning, model training, evaluation, and inference.
>>>>>>> f8e7041d6e1672244bf78f43f63bd7efc665b01d

---

## 📋 Project Overview

### What It Does

Predixa performs **two tasks simultaneously** on Amazon product reviews:

1. **Category Classification**: Predicts whether a review is for "All Beauty" or "Appliances" products
2. **Rating Prediction**: Predicts the star rating (1.0 – 5.0) based on review text

### Architecture

- **Model**: DistilBERT (distilbert-base-uncased)
- **Approach**: Multi-task learning with a shared DistilBERT backbone and two independent prediction heads
  - Category head: 2-class classification (CrossEntropyLoss)
  - Rating head: Regression (MSELoss)
- **Framework**: PyTorch with Hugging Face Transformers
- **Language Composition**: 58.9% Python | 41.1% HTML

---

## 🎯 Key Features

<<<<<<< HEAD
*   **[app.py](file:///c:/Users/dhana/ADIP/app.py)**: The main orchestrator file. It loads the raw datasets, reports pre- and post-cleaning record counts, executes the model training pipeline, and showcases sample predictions.
*   **[train_classifier.py](file:///c:/Users/dhana/ADIP/train_classifier.py)**: Contains the core dataset class (`ReviewDataset`), the PyTorch multi-task model definition (`MultiTaskDistilBERT`), and the custom training/evaluation loops.
*   **[predict.py](file:///c:/Users/dhana/ADIP/predict.py)**: CLI inference tool to load the trained model checkpoints and run predictions on arbitrary review strings.
*   **[model_service.py](file:///c:/Users/dhana/ADIP/model_service.py)**: Python microservice backend for high-performance PyTorch model inference.
*   **[server/](file:///c:/Users/dhana/ADIP/server/)**: Express API gateway (written in TypeScript) that handles requests and serves the frontend.
*   **[client/](file:///c:/Users/dhana/ADIP/client/)**: React + TypeScript single-page app (built with Vite) that provides a premium review intelligence dashboard.
*   **[data_functions.py](file:///c:/Users/dhana/ADIP/data_functions.py)**: Library for data cleaning steps (deduplication, whitespace trimming, lowercasing fields, and tracking statistics).
*   **[count_entries.py](file:///c:/Users/dhana/ADIP/count_entries.py)**: Out-of-core file streamer utility for checking records and inspecting metadata.
*   **`dataset/`** *(Ignored by Git)*: Folder hosting large raw JSON lines datasets (e.g., `All_Beauty.json`, `Appliances.json`).
*   **`distilbert_product_classifier/`** *(Ignored by Git)*: Destination folder for saving the trained model checkpoints (`model.pt`, configuration, and tokenizer).

---

## ⚡ Setup & Installation

### 1. Prerequisites & Dependencies
Ensure you have Python 3.8+ installed. Install the required machine learning packages:
```bash
pip install torch transformers pandas scikit-learn numpy flask

```

> [!TIP]
> **GPU Acceleration (Recommended)**: For significantly faster training, verify if you have an NVIDIA CUDA-enabled GPU. 
> To uninstall CPU-only PyTorch and set up CUDA support:
> ```bash
> pip uninstall torch
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```
> To verify CUDA is active in your terminal:
> ```bash
> python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
> ```

### 2. Dataset Preparation
Place the raw Amazon product reviews files inside the `dataset/` directory:
*   `dataset/All_Beauty.json`
*   `dataset/Appliances.json`

---

## 🚀 Running the Pipeline

### Option 1: Run the Full Pipeline
Executes the clean, train, evaluate, and test pipeline:
```bash
python app.py
```

### Option 2: Train the Model Separately
Runs the training loop and saves the best model to `./distilbert_product_classifier/`:
```bash
python train_classifier.py
```
*   *Note: Customize training size (`MAX_SAMPLES_PER_CLASS`), epochs, or learning rate directly within the config section in [train_classifier.py](file:///c:/Users/dhana/ADIP/train_classifier.py).*

### Option 3: Run Inference (CLI / Interactive)
To test predictions on arbitrary input texts using a saved checkpoint:
```bash
# Run with a custom review string:
python predict.py "This face cream is absolutely wonderful for dry skin, highly recommend!"

# Run interactive demo examples:
python predict.py
```

### Option 4: Run the Web UI Dashboard

To run the premium React web interface served by the Express backend and backed by the Python model microservice:

1. **Install Node.js dependencies**:
   ```bash
   npm run install:all
   ```

2. **Run all services concurrently (in development mode)**:
   ```bash
   npm run dev
   ```
   *This starts the Python model service on port `5001`, the Node.js backend on `5000`, and the React frontend on `5173`. Open `http://localhost:5173` to interact with the dashboard.*

3. **Build and run for production**:
   ```bash
   # Build client static bundle and compile TS server
   npm run build
   
   # Start the Python model service
   python model_service.py
   
   # Start the Express server (in a separate terminal)
   npm start --prefix server
   ```
   *Navigate to `http://localhost:5000` to access the production deployment.*
---

## 📊 Sample Pipeline Run

Below is a sample trace of the pipeline run on a GPU:

```text
Using device: cuda

============================================================
  Instance Counts — Before and After Cleaning
============================================================

  [All Beauty]
    Before cleaning : 371,345 instances
    After  cleaning : 362,713 instances
    Removed         : 8,632 instances

  [Appliances]
    Before cleaning : 602,777 instances
    After  cleaning : 591,371 instances
    Removed         : 11,406 instances

============================================================

=================================================================
  Training Multi-Task DistilBERT
  Epochs: 3  |  Batch size: 16  |  LR: 2e-05
=================================================================
  Epoch 1/3  | Train Loss: 1.4589  Acc: 71.84%  MAE: 0.658  | Val Loss: 0.8576  Acc: 84.00%  MAE: 0.447
    [OK] Best model saved (val acc: 84.00%)
  Epoch 2/3  | Train Loss: 0.6337  Acc: 85.34%  MAE: 0.353  | Val Loss: 0.8956  Acc: 85.88%  MAE: 0.457
  ...
```

---
## 🔒 Separation of Code and Data (Git Best Practices)

To avoid push limits and repository bloating, a custom `.gitignore` shields the codebase. Large raw datasets (`dataset/`) and model weight checkpoints (`distilbert_product_classifier/`) are excluded from Git tracking. This maintains a lightweight, production-ready code repository suitable for version control on GitHub.
=======
✅ **Data Cleaning**: Multi-step pipeline to handle missing values, duplicates, and text normalization  
✅ **Multi-Task Learning**: Jointly train category classification and rating prediction  
✅ **Model Checkpointing**: Saves the best model based on validation accuracy  
✅ **CLI Inference**: Predict category & rating from command-line text input  
✅ **Flexible Configuration**: Easy-to-modify hyperparameters for training  

---

## 📁 Project Structure
>>>>>>> f8e7041d6e1672244bf78f43f63bd7efc665b01d
