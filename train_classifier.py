"""
train_classifier.py — Multi-task DistilBERT model (PyTorch).

Tasks:
  1. Category classification : 0 = All Beauty, 1 = Appliances  (CrossEntropyLoss)
  2. Star-rating regression   : predict 1.0–5.0 from reviewText (MSELoss)

Input text : reviewText + [SEP] + summary  (from reviews JSON)
Labels     : label (int 0/1)  |  overall (float 1.0–5.0)
"""

import json
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- Reproducibility ----------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Config -------------------------------------------------------------------
DATASETS = [
    {"label": 0, "name": "All Beauty",  "reviews": "dataset/All_Beauty.json"},
    {"label": 1, "name": "Appliances",  "reviews": "dataset/Appliances.json"},
]

MAX_SAMPLES_PER_CLASS = 2_000    # ← increase for longer/better training (GPU recommended for >10k)
MAX_LEN               = 128
BATCH_SIZE            = 16
EPOCHS                = 3
LR                    = 2e-5
RATING_LOSS_WEIGHT    = 1.0      # λ  —  total_loss = cat_loss + λ * rating_loss
SAVE_DIR              = "./distilbert_product_classifier"
MODEL_NAME            = "distilbert-base-uncased"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- Data helpers -------------------------------------------------------------

def load_json_lines(filepath: str) -> list:
    """Load a JSON-lines file into a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_dataframe(datasets: list, max_per_class: int) -> pd.DataFrame:
    """
    Load reviews from each dataset, keep only rows with valid text and rating,
    sample up to max_per_class rows, assign category label.
    """
    frames = []
    for ds in datasets:
        print(f"\nLoading '{ds['name']}' from {ds['reviews']} …")
        rows = load_json_lines(ds["reviews"])
        df = pd.DataFrame(rows)

        # Keep rows with usable text and rating
        df["reviewText"] = df.get("reviewText", pd.Series(dtype=str)).fillna("")
        df["summary"]    = df.get("summary",    pd.Series(dtype=str)).fillna("")
        df["overall"]    = pd.to_numeric(df.get("overall", pd.Series(dtype=float)),
                                         errors="coerce")
        df = df.dropna(subset=["overall"])
        df = df[df["reviewText"].str.strip() != ""]

        # Combine text
        df["text"] = df["reviewText"].str.strip() + " [SEP] " + df["summary"].str.strip()

        # Sample
        if len(df) > max_per_class:
            df = df.sample(n=max_per_class, random_state=SEED)

        df["label"] = ds["label"]

        frames.append(df[["text", "overall", "label"]])
        print(f"  -> {len(df):,} samples kept")

    combined = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=SEED)
    print(f"\nTotal samples after combining: {len(combined):,}")
    return combined


# --- PyTorch Dataset ----------------------------------------------------------

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, ratings, tokenizer, max_len):
        self.texts    = texts
        self.labels   = labels
        self.ratings  = ratings
        self.tokenizer = tokenizer
        self.max_len  = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
            "rating":         torch.tensor(self.ratings[idx], dtype=torch.float),
        }


# --- Multi-Task Model ---------------------------------------------------------

class MultiTaskDistilBERT(nn.Module):
    """
    Shared DistilBERT backbone with two independent heads:
      • category_head : Linear(768 -> 2)  — BeautY vs Appliances
      • rating_head   : Linear(768 -> 1)  — predicted star rating 1.0–5.0
    """
    def __init__(self, model_name: str = MODEL_NAME, dropout: float = 0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        hidden = self.distilbert.config.dim   # 768 for base

        self.dropout       = nn.Dropout(dropout)
        self.category_head = nn.Linear(hidden, 2)
        self.rating_head   = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        # (batch, seq_len, hidden)
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation  ->  (batch, hidden)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Head 1 — category logits  (batch, 2)
        cat_logits = self.category_head(cls_output)

        # Head 2 — rating  (batch,)  scaled to 1–5 via sigmoid × 4 + 1
        raw_rating = self.rating_head(cls_output).squeeze(-1)
        rating_pred = torch.sigmoid(raw_rating) * 4.0 + 1.0

        return cat_logits, rating_pred


# --- Training & Evaluation ----------------------------------------------------

def compute_metrics(cat_logits, cat_labels, rating_preds, rating_targets):
    preds   = torch.argmax(cat_logits, dim=1).cpu().numpy()
    targets = cat_labels.cpu().numpy()
    acc     = accuracy_score(targets, preds)
    mae     = torch.mean(torch.abs(rating_preds - rating_targets)).item()
    return acc, mae


def train_one_epoch(model, loader, optimizer, ce_loss, mse_loss, device):
    model.train()
    total_loss = 0.0
    all_cat_preds, all_cat_labels = [], []
    all_rat_preds, all_rat_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        ratings        = batch["rating"].to(device)

        optimizer.zero_grad()
        cat_logits, rating_pred = model(input_ids, attention_mask)

        loss_cat    = ce_loss(cat_logits, labels)
        loss_rating = mse_loss(rating_pred, ratings)
        loss        = loss_cat + RATING_LOSS_WEIGHT * loss_rating

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_cat_preds.append(cat_logits.detach())
        all_cat_labels.append(labels.detach())
        all_rat_preds.append(rating_pred.detach())
        all_rat_labels.append(ratings.detach())

    cat_logits_all  = torch.cat(all_cat_preds)
    cat_labels_all  = torch.cat(all_cat_labels)
    rat_preds_all   = torch.cat(all_rat_preds)
    rat_labels_all  = torch.cat(all_rat_labels)

    avg_loss = total_loss / len(loader)
    acc, mae = compute_metrics(cat_logits_all, cat_labels_all,
                               rat_preds_all,  rat_labels_all)
    return avg_loss, acc, mae


@torch.no_grad()
def evaluate(model, loader, ce_loss, mse_loss, device):
    model.eval()
    total_loss = 0.0
    all_cat_preds, all_cat_labels = [], []
    all_rat_preds, all_rat_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        ratings        = batch["rating"].to(device)

        cat_logits, rating_pred = model(input_ids, attention_mask)

        loss_cat    = ce_loss(cat_logits, labels)
        loss_rating = mse_loss(rating_pred, ratings)
        loss        = loss_cat + RATING_LOSS_WEIGHT * loss_rating

        total_loss += loss.item()
        all_cat_preds.append(cat_logits)
        all_cat_labels.append(labels)
        all_rat_preds.append(rating_pred)
        all_rat_labels.append(ratings)

    cat_logits_all = torch.cat(all_cat_preds)
    cat_labels_all = torch.cat(all_cat_labels)
    rat_preds_all  = torch.cat(all_rat_preds)
    rat_labels_all = torch.cat(all_rat_labels)

    avg_loss = total_loss / len(loader)
    acc, mae = compute_metrics(cat_logits_all, cat_labels_all,
                               rat_preds_all,  rat_labels_all)
    return avg_loss, acc, mae

# --- Show sample predictions -------------------------------------------------

@torch.no_grad()
def show_test_predictions(model, val_df, tokenizer, cfg, n=10):
    """Print predictions vs ground truth for n random samples from the val set."""
    categories = cfg["categories"]
    sample = val_df.sample(n=n, random_state=SEED).reset_index(drop=True)

    model.eval()
    print("\n" + "=" * 70)
    print(f"  Sample Predictions on Test Set  ({n} instances)")
    print("=" * 70)
    header = (f"{'#':>2}  {'True Category':<15} {'Pred Category':<15}"
              f"  {'Conf':>6}  {'True *':>6}  {'Pred *':>6}  Review (truncated)")
    print(header)
    print("-" * 70)

    for i, row in sample.iterrows():
        enc = tokenizer(
            row["text"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        cat_logits, rating_pred = model(input_ids, attention_mask)
        probs      = torch.softmax(cat_logits, dim=1).squeeze(0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item() * 100
        pred_rating = rating_pred.item()

        true_cat  = categories[str(row["label"])]
        pred_cat  = categories[str(pred_class)]
        true_star = row["overall"]
        snippet   = row["text"][:45].replace("\n", " ")

        correct_marker = "[OK]" if pred_class == row["label"] else "[X]"
        print(f"{i+1:>2}{correct_marker} {true_cat:<15} {pred_cat:<15}"
              f"  {confidence:>5.1f}%  {true_star:>6.1f}  {pred_rating:>6.2f}  {snippet}…")

    print("=" * 70)


def main():
    # 1. Load & prepare data --------------------------------------------------
    df = build_dataframe(DATASETS, MAX_SAMPLES_PER_CLASS)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED,
                                        stratify=df["label"])
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    # 2. Tokenizer ------------------------------------------------------------
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # 3. Datasets & DataLoaders -----------------------------------------------
    def make_ds(df_split):
        return ReviewDataset(
            texts    = df_split["text"].tolist(),
            labels   = df_split["label"].tolist(),
            ratings  = df_split["overall"].tolist(),
            tokenizer = tokenizer,
            max_len   = MAX_LEN,
        )

    train_loader = DataLoader(make_ds(train_df), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(make_ds(val_df),   batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=0)

    # 4. Model ----------------------------------------------------------------
    print(f"\nLoading model: {MODEL_NAME}")
    model = MultiTaskDistilBERT(MODEL_NAME).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    # 5. Training loop --------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Training Multi-Task DistilBERT")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LR}"
          f"  |  Device: {DEVICE}")
    print("=" * 65)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_mae = train_one_epoch(
            model, train_loader, optimizer, ce_loss, mse_loss, DEVICE)
        vl_loss, vl_acc, vl_mae = evaluate(
            model, val_loader,   ce_loss, mse_loss, DEVICE)

        print(f"Epoch {epoch}/{EPOCHS}"
              f" | Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%  MAE: {tr_mae:.3f}"
              f" | Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}%  MAE: {vl_mae:.3f}")

        # Save best model checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
            print(f"  [OK] Best model saved (val acc: {best_val_acc*100:.2f}%)")

    # 6. Save tokenizer alongside model ---------------------------------------
    tokenizer.save_pretrained(SAVE_DIR)

    # Save config for predict.py
    config = {"model_name": MODEL_NAME, "max_len": MAX_LEN,
               "categories": {"0": "All Beauty", "1": "Appliances"}}
    import json as _json
    with open(os.path.join(SAVE_DIR, "task_config.json"), "w") as f:
        _json.dump(config, f, indent=2)

    print(f"\n[OK] Training complete. Best val accuracy: {best_val_acc*100:.2f}%")
    print(f"[OK] Model + tokenizer saved to: {SAVE_DIR}/")

    # 7. Load best model & show predictions on 10 test instances --------------
    print("\nLoading best saved model for test predictions…")
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "model.pt"),
                                     map_location=DEVICE))
    show_test_predictions(model, val_df, tokenizer, config, n=10)

    print(f"\nRun inference on any text with:\n  python predict.py \"<your review text here>\"")


if __name__ == "__main__":
    main()
