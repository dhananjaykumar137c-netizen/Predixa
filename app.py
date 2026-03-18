"""
app.py — Main entry point for the ADIP project.

Pipeline:
  1. Print instance counts for both raw datasets (before cleaning).
  2. Clean both datasets and print counts after cleaning.
  3. Build combined DataFrame using train_classifier helpers, train the
     DistilBERT multi-task model on 80% of the data.
  4. Predict category + rating for 10 random instances from the 20% test split.
"""

import io
import os
import json
import contextlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast

from data_functions import clean_dataset

# ── Reuse helpers from train_classifier.py ────────────────────────────────────
from train_classifier import (
    load_json_lines,
    build_dataframe,
    ReviewDataset,
    MultiTaskDistilBERT,
    train_one_epoch,
    evaluate,
    show_test_predictions,
    DATASETS,
    MAX_SAMPLES_PER_CLASS,
    MAX_LEN,
    BATCH_SIZE,
    EPOCHS,
    LR,
    SAVE_DIR,
    MODEL_NAME,
    DEVICE,
    SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def count_json_lines(filepath: str) -> int:
    """Count records in a JSON-lines file without loading it fully."""
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 & 2 — Instance counts before and after cleaning
# ─────────────────────────────────────────────────────────────────────────────

def report_instance_counts():
    """Print instance counts for each dataset before and after cleaning."""
    print("\n" + "=" * 60)
    print("  Instance Counts — Before and After Cleaning")
    print("=" * 60)

    for ds in DATASETS:
        filepath = ds["reviews"]
        label    = ds["name"]

        before = count_json_lines(filepath)

        # Suppress verbose step-by-step output from clean_dataset
        with contextlib.redirect_stdout(io.StringIO()):
            cleaned_df = clean_dataset(load_json_lines(filepath))

        after = cleaned_df.shape[0]

        print(f"\n  [{label}]")
        print(f"    Before cleaning : {before:,} instances")
        print(f"    After  cleaning : {after:,} instances")
        print(f"    Removed         : {before - after:,} instances")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build data and train the model
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    """
    Build the combined DataFrame from both datasets, do an 80/20 split,
    train the multi-task DistilBERT model, and save the best checkpoint.

    Returns: model, tokenizer, config dict, val_df (the 20% test split)
    """
    # Build combined dataframe (suppress verbose loading prints)
    with contextlib.redirect_stdout(io.StringIO()):
        df = build_dataframe(DATASETS, MAX_SAMPLES_PER_CLASS)

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    print(f"\n  Train split : {len(train_df):,} instances  (80%)")
    print(f"  Test  split : {len(val_df):,} instances  (20%)")

    # Load tokenizer
    print(f"\n  Loading tokenizer : {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def make_dataset(df_split):
        return ReviewDataset(
            texts     = df_split["text"].tolist(),
            labels    = df_split["label"].tolist(),
            ratings   = df_split["overall"].tolist(),
            tokenizer = tokenizer,
            max_len   = MAX_LEN,
        )

    train_loader = DataLoader(make_dataset(train_df), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(make_dataset(val_df),   batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=0)

    # Initialise model
    print(f"  Loading model     : {MODEL_NAME}  |  Device: {DEVICE}")
    model     = MultiTaskDistilBERT(MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    # Training loop
    print("\n" + "=" * 65)
    print("  Training Multi-Task DistilBERT")
    print(f"  Epochs: {EPOCHS}  |  Batch size: {BATCH_SIZE}  |  LR: {LR}")
    print("=" * 65)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_mae = train_one_epoch(
            model, train_loader, optimizer, ce_loss, mse_loss, DEVICE)
        vl_loss, vl_acc, vl_mae = evaluate(
            model, val_loader, ce_loss, mse_loss, DEVICE)

        print(f"  Epoch {epoch}/{EPOCHS}"
              f"  | Train  Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%  MAE: {tr_mae:.3f}"
              f"  | Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}%  MAE: {vl_mae:.3f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))
            print(f"    [OK] Best model saved  (val acc: {best_val_acc*100:.2f}%)")

    # Save tokenizer + config so predict.py can reload the model later
    tokenizer.save_pretrained(SAVE_DIR)
    config = {
        "model_name": MODEL_NAME,
        "max_len":    MAX_LEN,
        "categories": {"0": "All Beauty", "1": "Appliances"},
    }
    with open(os.path.join(SAVE_DIR, "task_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  [OK] Training complete.  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  [OK] Model saved to      : {SAVE_DIR}/")

    # Reload the best checkpoint before returning
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "model.pt"),
                   map_location=DEVICE, weights_only=True)
    )
    model.eval()

    return model, tokenizer, config, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Predict on 10 random test instances
# ─────────────────────────────────────────────────────────────────────────────

def predict_test_instances(model, val_df, tokenizer, config):
    """
    Show predicted category + rating for 10 random instances from the 20%
    test split using show_test_predictions() from train_classifier.py.

    Output format per instance:
        #  True Category   Pred Category   Conf    True ★  Pred ★  Review
    """
    show_test_predictions(model, val_df, tokenizer, config, n=10)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Step 1 & 2 — Before / after cleaning counts
    report_instance_counts()

    # Step 3 — Train model on 80% of data
    print("\n" + "=" * 60)
    print("  Model Training  (80% Train / 20% Test)")
    print("=" * 60)
    model, tokenizer, config, val_df = train_model()

    # Step 4 — Predict 10 random instances from the 20% test set
    predict_test_instances(model, val_df, tokenizer, config)


if __name__ == "__main__":
    main()
