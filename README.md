# Predixa

**Predixa** is a multi-task deep learning project that classifies product reviews and predicts star ratings using a fine-tuned **DistilBERT** model. It demonstrates end-to-end NLP workflows including data cleaning, model training, evaluation, and inference.

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

✅ **Data Cleaning**: Multi-step pipeline to handle missing values, duplicates, and text normalization  
✅ **Multi-Task Learning**: Jointly train category classification and rating prediction  
✅ **Model Checkpointing**: Saves the best model based on validation accuracy  
✅ **CLI Inference**: Predict category & rating from command-line text input  
✅ **Flexible Configuration**: Easy-to-modify hyperparameters for training  

---

## 📁 Project Structure
