import json
from collections import Counter
from PIL import Image
from sklearn.model_selection import train_test_split
import config

EXTS = {".jpg", ".jpeg", ".png"}


def collect_items():
    items, bad = [], []
    for folder, label in config.CLASS_FOLDERS.items():
        for p in sorted((config.DATA_DIR / folder).iterdir()):
            if p.suffix.lower() not in EXTS:
                continue
            try:
                with Image.open(p) as im:
                    im.verify()                       # detects corrupt files
            except Exception:
                bad.append(str(p))
                continue
            items.append([str(p.relative_to(config.DATA_DIR)), label])
    return items, bad


def make_split(items):
    """Create the deterministic stratified train/validation/test split"""
    labels = [label for _, label in items]
    test_val = config.SPLIT["val"] + config.SPLIT["test"]
    train, rest = train_test_split(
        items, test_size=test_val, stratify=labels, random_state=config.SEED)
    val, test = train_test_split(
        rest, test_size=config.SPLIT["test"] / test_val,
        stratify=[label for _, label in rest], random_state=config.SEED)
    return {"train": train, "val": val, "test": test}


def main():
    items, bad = collect_items()
    print(f"Found {len(items)} valid images, {len(bad)} corrupt (skipped).")
    for b in bad:
        print("  corrupt:", b)
    split = make_split(items)
    with open(config.SPLIT_FILE, "w") as f:
        json.dump(split, f)
    for k, v in split.items():
        c = Counter(l for _, l in v)
        print(f"{k:5s}: {len(v):4d} images  (cats={c[0]}, dogs={c[1]})")


if __name__ == "__main__":
    main()