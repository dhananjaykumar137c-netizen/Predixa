"""
predict.py — Run inference with the saved multi-task DistilBERT model.

Usage:
    python predict.py "This face cream is absolutely wonderful for dry skin"
    python predict.py  (runs built-in demo examples)
"""

import sys
import os
import json
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel

SAVE_DIR = "./distilbert_product_classifier"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Must mirror the model class in train_classifier.py ───────────────────────
class MultiTaskDistilBERT(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.3):
        super().__init__()
        self.distilbert     = DistilBertModel.from_pretrained(model_name)
        hidden              = self.distilbert.config.dim
        self.dropout        = nn.Dropout(dropout)
        self.category_head  = nn.Linear(hidden, 2)
        self.rating_head    = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs    = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        cat_logits = self.category_head(cls_output)
        raw_rating = self.rating_head(cls_output).squeeze(-1)
        rating_pred = torch.sigmoid(raw_rating) * 4.0 + 1.0
        return cat_logits, rating_pred


def load_model():
    config_path = os.path.join(SAVE_DIR, "task_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No trained model found at '{SAVE_DIR}'.\n"
            "Please run  python train_classifier.py  first."
        )

    with open(config_path) as f:
        cfg = json.load(f)

    tokenizer  = DistilBertTokenizerFast.from_pretrained(SAVE_DIR)
    model      = MultiTaskDistilBERT(cfg["model_name"]).to(DEVICE)

    # FIX: weights_only=True avoids deprecation warning in PyTorch 2.x
    state_dict = torch.load(os.path.join(SAVE_DIR, "model.pt"),
                            map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer, cfg


@torch.no_grad()
def predict(text: str, model, tokenizer, cfg, max_len: int = 128):
    enc = tokenizer(
        text,
        max_length    = max_len,
        padding       = "max_length",
        truncation    = True,
        return_tensors = "pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    cat_logits, rating_pred = model(input_ids, attention_mask)

    probs      = torch.softmax(cat_logits, dim=1).squeeze(0)
    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item() * 100
    rating     = rating_pred.item()
    categories = cfg["categories"]

    return {
        "category"   : categories[str(pred_class)],
        "label"      : pred_class,
        "confidence" : confidence,
        "rating"     : round(rating, 2),
    }


def print_result(text: str, result: dict):
    print("\n" + "─" * 55)
    print(f"  Input    : {text[:80]}{'…' if len(text) > 80 else ''}")
    print(f"  Category : {result['category']}  (confidence: {result['confidence']:.1f}%)")
    print(f"  Rating   : {result['rating']:.1f} / 5.0  {'★' * round(result['rating'])}")
    print("─" * 55)


DEMO_EXAMPLES = [
    "This moisturizer is absolutely amazing for dry skin. It leaves a beautiful glow!",
    "The washing machine stopped working after just 2 months. Very disappointed with the build quality.",
    "Love this lipstick shade! It stays on all day and the packaging is gorgeous.",
    "The toaster heats evenly and the slots are wide enough for thick bread slices. Very happy!",
]


def main():
    model, tokenizer, cfg = load_model()
    print(f"Model loaded from '{SAVE_DIR}' | Device: {DEVICE}")

    if len(sys.argv) > 1:
        text   = " ".join(sys.argv[1:])
        result = predict(text, model, tokenizer, cfg)
        print_result(text, result)
    else:
        print("\nNo input provided — running demo examples:\n")
        for text in DEMO_EXAMPLES:
            result = predict(text, model, tokenizer, cfg)
            print_result(text, result)


if __name__ == "__main__":
    main()