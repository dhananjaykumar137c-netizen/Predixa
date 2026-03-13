import json

# ── Print all field/column names from each dataset ──────────────────────────

# Field names from All_Beauty.json (reviews)
with open("dataset/All_Beauty.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            first_review = json.loads(line)
            break

print("=" * 50)
print("Fields in All_Beauty.json (reviews):")
print("=" * 50)
for i, field in enumerate(first_review.keys(), start=1):
    print(f"  {i}. {field}")

# Field names from meta_All_Beauty.json (metadata)
with open("dataset/meta_All_Beauty.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            first_meta = json.loads(line)
            break

print()
print("=" * 50)
print("Fields in meta_All_Beauty.json (metadata):")
print("=" * 50)
for i, field in enumerate(first_meta.keys(), start=1):
    print(f"  {i}. {field}")

print()
print("=" * 50)
print()

# ── Original analysis ────────────────────────────────────────────────────────

# Load All_Beauty.json (reviews) — build a mapping: asin -> list of review entries
reviews_by_asin = {}
with open("dataset/All_Beauty.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        asin = entry.get("asin")
        if asin:
            reviews_by_asin.setdefault(asin, []).append(entry)

# Load first 100 entries from meta_All_Beauty.json
meta_entries = []
with open("dataset/meta_All_Beauty.json", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        meta_entries.append(json.loads(line))
        if len(meta_entries) == 100:
            break

print(f"{'#':<5} {'ASIN':<15} {'Title (truncated)':<45} {'Review Count'}")
print("-" * 80)

matched = 0
unmatched = 0

for i, meta in enumerate(meta_entries, start=1):
    asin = meta.get("asin", "N/A")
    title = meta.get("title", "")[:45]
    count = len(reviews_by_asin.get(asin, []))
    if count > 0:
        matched += 1
    else:
        unmatched += 1
    print(f"{i:<5} {asin:<15} {title:<45} {count}")

print("-" * 80)
print(f"\nSummary (first 100 meta entries):")
print(f"  Total meta entries processed : 100")
print(f"  Entries WITH reviews         : {matched}")
print(f"  Entries WITHOUT reviews      : {unmatched}")
print(f"  Total review records matched : {sum(len(reviews_by_asin.get(m.get('asin',''), [])) for m in meta_entries)}")
