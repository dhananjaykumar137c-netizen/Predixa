"""
app.py — Main entry point for the ADIP project.
All dataset utility functions are imported from data_functions.py.
"""

import json
from data_functions import print_field_names, print_record_count, clean_dataset
from sklearn.model_selection import train_test_split


def load_json_lines(filepath: str) -> list:
    """Loads a JSON-lines file into a list of dicts."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def process_dataset(label: str, reviews_file: str, meta_file: str) -> dict:
    """Run the full pipeline for one category (reviews + meta)."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Field names & record counts
    print_field_names(reviews_file)
    print_field_names(meta_file)
    print_record_count(reviews_file)
    print_record_count(meta_file)

    # Clean
    reviews_clean = clean_dataset(load_json_lines(reviews_file))
    meta_clean    = clean_dataset(load_json_lines(meta_file))

    # Train / test split
    reviews_train, reviews_test = train_test_split(
        reviews_clean, test_size=0.2, random_state=42
    )
    meta_train, meta_test = train_test_split(
        meta_clean, test_size=0.2, random_state=42
    )

    print(f"Reviews  — Train: {reviews_train.shape[0]:>7,} rows x {reviews_train.shape[1]} cols")
    print(f"Reviews  — Test : {reviews_test.shape[0]:>7,} rows x {reviews_test.shape[1]} cols")
    print(f"Metadata — Train: {meta_train.shape[0]:>7,} rows x {meta_train.shape[1]} cols")
    print(f"Metadata — Test : {meta_test.shape[0]:>7,} rows x {meta_test.shape[1]} cols")

    # Field names from cleaned DataFrames
    print_field_names(reviews_clean)
    print_field_names(meta_clean)

    return {
        "reviews_train": reviews_train, "reviews_test": reviews_test,
        "meta_train":    meta_train,    "meta_test":    meta_test,
    }


def main():
    datasets = [
        ("All Beauty",  "dataset/All_Beauty.json",   "dataset/meta_All_Beauty.json"),
        ("Appliances",  "dataset/Appliances.json",    "dataset/meta_Appliances.json"),
    ]

    results = {}
    for label, reviews_file, meta_file in datasets:
        results[label] = process_dataset(label, reviews_file, meta_file)

    print(f"\nTotal number of elements in results: {len(results)}")


if __name__ == "__main__":
    main()
