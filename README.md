<<<<<<< HEAD
# ADIP: Alternative Data Intelligence Pipeline 🚀

Welcome to **ADIP** (Alternative Data Intelligence Pipeline), a machine learning pipeline designed to clean, process, and analyze large e-commerce product datasets, and train a **multi-task PyTorch model** using a shared `distilbert-base-uncased` backbone.
=======
# PREDIXA 🚀

Welcome to **PREDIXA** , a machine learning pipeline designed to clean, process, and analyze large e-commerce product datasets, and train a **multi-task PyTorch model** using a shared `distilbert-base-uncased` backbone.
>>>>>>> 6cd6e2bdeef949413a547bea1556f25d0276c683

The pipeline performs two prediction tasks simultaneously based on product reviews (concatenating the review body and summary):
1. **Category Classification** (Binary classification): Classifies reviews into **All Beauty** (0) or **Appliances** (1).
2. **Rating Prediction** (Regression): Predicts a star rating from **1.0 to 5.0**.

---

## 🏗️ Neural Network Architecture

The architecture uses a shared pretrained **DistilBERT** encoder to extract text representations, feeding the `[CLS]` token embedding into two independent linear output heads:

```
  [Input Text (Review + Summary)]
                 │
      DistilBERT Shared Backbone
                 │
       [CLS] Token Embedding (768-dim)
              ╱ ╲
             ╱   ╲
   Task Head 1    Task Head 2
   (Linear 768→2)  (Linear 768→1)
         │              │
    Category logits  Star Rating
    (CrossEntropy)   (Sigmoid × 4 + 1)
         │              │
    [All Beauty    [Rating 1.0–5.0]
     vs Appliances]
```

The model is optimized using a weighted sum of losses:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{category}} + \lambda \cdot \mathcal{L}_{\text{rating}}$$
* Where $\mathcal{L}_{\text{category}}$ is Cross-Entropy Loss, $\mathcal{L}_{\text{rating}}$ is Mean Squared Error (MSE), and $\lambda = 1.0$ (hyperparameter).

---

## 📂 Repository Structure

*   **[app.py](file:///c:/Users/dhana/ADIP/app.py)**: The main orchestrator file. It loads the raw datasets, reports pre- and post-cleaning record counts, executes the model training pipeline, and showcases sample predictions.
*   **[train_classifier.py](file:///c:/Users/dhana/ADIP/train_classifier.py)**: Contains the core dataset class (`ReviewDataset`), the PyTorch multi-task model definition (`MultiTaskDistilBERT`), and the custom training/evaluation loops.
*   **[predict.py](file:///c:/Users/dhana/ADIP/predict.py)**: CLI inference tool to load the trained model checkpoints and run predictions on arbitrary review strings.
<<<<<<< HEAD
*   **[web_app.py](file:///c:/Users/dhana/ADIP/web_app.py)**: Flask web server backend linking the multi-task model with the user interface.
*   **[templates/index.html](file:///c:/Users/dhana/ADIP/templates/index.html)**: Interactive, glassmorphic single-page web dashboard for inputting reviews and displaying results.
=======
>>>>>>> 6cd6e2bdeef949413a547bea1556f25d0276c683
*   **[data_functions.py](file:///c:/Users/dhana/ADIP/data_functions.py)**: Library for data cleaning steps (deduplication, whitespace trimming, lowercasing fields, and tracking statistics).
*   **[count_entries.py](file:///c:/Users/dhana/ADIP/count_entries.py)**: Out-of-core file streamer utility for checking records and inspecting metadata.
*   **`dataset/`** *(Ignored by Git)*: Folder hosting large raw JSON lines datasets (e.g., `All_Beauty.json`, `Appliances.json`).
*   **`distilbert_product_classifier/`** *(Ignored by Git)*: Destination folder for saving the trained model checkpoints (`model.pt`, configuration, and tokenizer).

---

## ⚡ Setup & Installation

### 1. Prerequisites & Dependencies
Ensure you have Python 3.8+ installed. Install the required machine learning packages:
```bash
<<<<<<< HEAD
pip install torch transformers pandas scikit-learn numpy flask
=======
pip install torch transformers pandas scikit-learn numpy
>>>>>>> 6cd6e2bdeef949413a547bea1556f25d0276c683
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

<<<<<<< HEAD
### Option 4: Run the Web UI Dashboard
To run the interactive, premium web interface:
```bash
python web_app.py
```
*   *Navigate to `http://127.0.0.1:5000` in your web browser.*

=======
>>>>>>> 6cd6e2bdeef949413a547bea1556f25d0276c683
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
<<<<<<< HEAD

## 🔒 Separation of Code and Data (Git Best Practices)

To avoid push limits and repository bloating, a custom `.gitignore` shields the codebase. Large raw datasets (`dataset/`) and model weight checkpoints (`distilbert_product_classifier/`) are excluded from Git tracking. This maintains a lightweight, production-ready code repository suitable for version control on GitHub.
=======
>>>>>>> 6cd6e2bdeef949413a547bea1556f25d0276c683
