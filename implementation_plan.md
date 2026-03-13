# Multi-Task DistilBERT Product Model

Build a **multi-task PyTorch model** using `distilbert-base-uncased` as a shared backbone with **two independent output heads**:

| Task | Type | Target | Output |
|---|---|---|---|
| **Task 1** | Binary Classification | Category | 0 = Beauty, 1 = Appliances |
| **Task 2** | Regression | Star Rating | Predicted 1.0 – 5.0 |

**Input**: `reviewText` + `summary` concatenated (from the reviews dataset).  
**Label source**: `label` (category) and `overall` (star rating 1–5) columns.

---

## Architecture

```
  [Input Text]
       │
  DistilBERT Backbone  (shared weights)
       │
  [CLS] token embedding  (768-dim)
      ╱ ╲
     ╱   ╲
Head 1    Head 2
(Linear   (Linear
 768→2)    768→1)
   │          │
Category   Rating
(Softmax)  (Sigmoid
            × 4 + 1)
```

The total loss is a **weighted sum**: `loss = loss_category + λ × loss_rating`
- `loss_category` → CrossEntropyLoss
- `loss_rating` → MSELoss
- `λ = 1.0` (tunable)

---

## User Review Required

> [!IMPORTANT]
> **Sample size**: Full data = ~763K rows. Training on all of it takes many hours on CPU. Default cap is **20,000 per class (40K total)**. Adjust `MAX_SAMPLES_PER_CLASS` in the script if you want more/less.

> [!WARNING]
> **Dependencies needed** (if not installed):
> ```
> pip install transformers torch scikit-learn
> ```

---

## Proposed Changes

### [NEW] [train_classifier.py](file:///c:/Users/dhana/ADIP/train_classifier.py)

Standalone PyTorch training script:

1. **Data prep** — Load + clean JSON files using existing [load_json_lines](file:///c:/Users/dhana/ADIP/app.py#11-20) + [clean_dataset](file:///c:/Users/dhana/ADIP/data_functions.py#69-189). Add `label` column (0/1). Use `overall` as the rating target. Concatenate `reviewText` + `summary`.
2. **Dataset class** — Custom `torch.utils.data.Dataset` that tokenizes text with `DistilBertTokenizerFast`.
3. **Model class** — `MultiTaskDistilBERT(nn.Module)` with DistilBERT backbone + two heads.
4. **Training loop** — Pure PyTorch loop (no Trainer API). Trains for N epochs with `AdamW`.
5. **Evaluation** — Per-epoch accuracy (category) + MAE (rating) on val split.
6. **Save** — Saves `model.pt` (state dict) + tokenizer to `./distilbert_product_classifier/`.

Hyperparameters:
| Param | Default |
|---|---|
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Max token length | 128 |
| Max samples/class | 20,000 |
| Rating loss weight λ | 1.0 |

---

## Verification Plan

### Run Training
```
python train_classifier.py
```
Expected output per epoch:
```
Epoch 1/3 | Loss: X.XX | Cat Acc: XX.X% | Rating MAE: X.XX
```

### Prediction Test
```
python predict.py "This moisturizer left my skin glowing and soft"
```
Expected:
```
Category : Beauty (confidence: 97.3%)
Rating   : 4.6 / 5.0
```

### [NEW] [predict.py](file:///c:/Users/dhana/ADIP/predict.py)
Small CLI script to load the saved model and run inference on any input string.
