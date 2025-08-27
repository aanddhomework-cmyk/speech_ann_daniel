# @title splitting train-val
from pathlib import Path
import zipfile, json, random

EMB_ROOT = Path("embedding_batches_zip")   # folder with batch_000.zip, batch_001.zip, ...
VAL_FRAC = 0.20
SEED = 42

assert EMB_ROOT.exists(), f"Missing folder: {EMB_ROOT.resolve()}"
zips = sorted(EMB_ROOT.glob("batch_*.zip"))
assert zips, f"No zip files found in {EMB_ROOT.resolve()}"

# Index all utterances: arcname -> zip filename, and group by speaker
by_spk = {}
for zp in zips:
    with zipfile.ZipFile(zp, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".pt"):
                continue
            spk = name.split("/", 1)[0]  # "id10001/..." -> "id10001"
            by_spk.setdefault(spk, []).append({"zip": zp.name, "name": name, "spk": spk})

speakers = sorted(by_spk.keys())
print(f"Speakers: {len(speakers)}")

# Closed-set label map (all speakers get a class id)
label_map = {spk: i for i, spk in enumerate(speakers)}

# Per-speaker utterance split
random.seed(SEED)
train_items, val_items = [], []
for spk in speakers:
    items = by_spk[spk][:]
    random.shuffle(items)
    k = max(1, int(len(items) * VAL_FRAC))  # at least 1 utt in val if possible
    val_items.extend(items[:k])
    train_items.extend(items[k:])

print(f"Utterances: train={len(train_items)}  val={len(val_items)}")

# Write JSON with everything needed for dataset loading
Path("splits").mkdir(exist_ok=True)
with open("splits/closed_set_splits.json", "w") as f:
    json.dump({
        "label_map": label_map,
        "train": train_items,
        "val": val_items
    }, f)

print("Wrote splits/closed_set_splits.json")