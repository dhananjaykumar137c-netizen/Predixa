import json
import pandas as pd


def print_field_names(source) -> None:
    """
    Prints all field/column names from a dataset source.

    Parameters
    ----------
    source : str or pd.DataFrame
        Either a path to a JSON-lines file (str), or a pandas DataFrame
        (e.g. the output of clean_dataset()).
    """
    if isinstance(source, pd.DataFrame):
        # ── DataFrame input (cleaned dataset) ────────────────────────────────
        fields = list(source.columns)
        label = "DataFrame (cleaned dataset)"
    else:
        # ── File path input ───────────────────────────────────────────────────
        first_entry = None
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    first_entry = json.loads(line)
                    break

        if first_entry is None:
            print(f"No data found in '{source}'.")
            return

        fields = list(first_entry.keys())
        label = source

    print("=" * 50)
    print(f"Field names in: {label}")
    print(f"Total fields  : {len(fields)}")
    print("=" * 50)
    for i, field in enumerate(fields, start=1):
        print(f"  {i:>3}. {field}")
    print()


def print_record_count(filepath: str) -> None:
    """
    Counts and prints the total number of records (entries) in a JSON-lines file.
    Works with any JSON-lines dataset — does NOT load the whole file into memory.

    Parameters
    ----------
    filepath : str
        Path to a JSON-lines file (one JSON object per line).
    """
    count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    print("=" * 50)
    print(f"Record count in: {filepath}")
    print("=" * 50)
    print(f"  Total records : {count:,}")
    print()


def clean_dataset(data: list, key_id: str = "asin") -> pd.DataFrame:
    """
    Cleans a dataset (list of dicts) by applying five steps:

      1. Drop all-NaN columns and all-blank rows.
      2. Drop exact duplicate rows.
      3. Drop rows missing the key ID field ('asin').
      4. Standardize text case: lowercase all plain string columns
         (e.g. brand, category) so 'Adidas'/'ADIDAS'/'adidas' merge.
      5. Strip leading/trailing whitespace from plain string columns.
         (Does NOT touch sentences or columns with embedded whitespace.)

    Parameters
    ----------
    data : list of dict
        The dataset loaded from a JSON-lines file (list of records).
    key_id : str, optional
        The column name used as the primary/join key. Rows with a
        missing value in this column are dropped. Default: 'asin'.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame after all five steps.
    """
    df = pd.DataFrame(data)

    original_rows, original_cols = df.shape
    print("=" * 60)
    print("  Dataset Cleaning Report")
    print("=" * 60)
    print(f"  Original shape : {original_rows:,} rows x {original_cols} columns")
    print()

    # ── Step 1: Drop all-NaN columns and all-blank rows ──────────────────────
    all_nan_cols = [col for col in df.columns if df[col].isna().all()]
    df.drop(columns=all_nan_cols, inplace=True)

    all_blank_rows_before = len(df)
    df.dropna(how="all", inplace=True)
    blank_rows_dropped = all_blank_rows_before - len(df)

    print("  [Step 1] All-NaN / All-blank cleanup")
    if all_nan_cols:
        print(f"    Columns dropped (100% NaN) : {len(all_nan_cols)}")
        for col in all_nan_cols:
            print(f"      - {col}")
    else:
        print("    Columns dropped (100% NaN) : 0  (none found)")
    print(f"    Rows dropped   (all blank) : {blank_rows_dropped:,}")
    print()

    # ── Build a hashable comparison DataFrame ────────────────────────────────
    # Some columns may hold unhashable types (list, dict).
    # We stringify those cells in a temporary copy used for Step 2 (dedup) only.
    df.reset_index(drop=True, inplace=True)
    df_cmp = df.copy()
    for col in df_cmp.columns:
        if df_cmp[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_cmp[col] = df_cmp[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )

    # ── Step 2: Drop exact duplicate rows ────────────────────────────────────
    before_dedup = len(df)
    duplicate_mask = df_cmp.duplicated(keep="first")
    df = df[~duplicate_mask].reset_index(drop=True)
    df_cmp = df_cmp[~duplicate_mask].reset_index(drop=True)
    duplicates_dropped = before_dedup - len(df)

    print("  [Step 2] Exact duplicate rows")
    print(f"    Duplicate rows dropped     : {duplicates_dropped:,}")
    print()

    # ── Step 3: Drop rows missing the key ID ────────────────────────────────
    before_key_drop = len(df)
    if key_id in df.columns:
        df.dropna(subset=[key_id], inplace=True)
        df = df[df[key_id].astype(str).str.strip() != ""].reset_index(drop=True)
    key_rows_dropped = before_key_drop - len(df)

    print(f"  [Step 3] Rows missing key ID ('{key_id}')")
    print(f"    Rows dropped   (no key ID) : {key_rows_dropped:,}")
    print()

    # ── Step 4 & 5: Lowercase + strip whitespace on plain string columns ─────
    # A "plain string" column: dtype is object AND no cell holds a list/dict.
    # This avoids mangling multi-word sentences or nested structures.
    plain_str_cols = [
        col for col in df.columns
        if df[col].dtype == object
        and not df[col].dropna().apply(lambda x: isinstance(x, (list, dict))).any()
    ]

    for col in plain_str_cols:
        df[col] = df[col].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

    print(f"  [Step 4] Standardize text case (lowercase)")
    print(f"    Plain string columns updated : {len(plain_str_cols)}")
    for col in plain_str_cols:
        print(f"      - {col}")
    print()

    print(f"  [Step 5] Strip leading/trailing whitespace")
    print(f"    Plain string columns updated : {len(plain_str_cols)}")
    print(f"    (same columns as Step 4 — applied in one pass)")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    final_rows, final_cols = df.shape
    print("  " + "-" * 56)
    print(f"  Final shape    : {final_rows:,} rows x {final_cols} columns")
    print(f"  Rows removed   : {original_rows - final_rows:,}")
    print(f"  Cols removed   : {original_cols - final_cols}")
    print("=" * 60)
    print()

    return df


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_field_names("dataset/All_Beauty.json")
    print_field_names("dataset/meta_All_Beauty.json")
    print_record_count("dataset/All_Beauty.json")
    print_record_count("dataset/meta_All_Beauty.json")
