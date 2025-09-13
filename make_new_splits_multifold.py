from pathlib import Path
import zipfile, json, random, math
from typing import List, Dict

#config:
ROOTS: List[Path] = [Path("embedding_batches_zip")]
SPLITS_DIR = Path("splits")

SEED_BASE = 42
K_FOLDS   = 5

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
TEST_FRAC  = 0.10
CALIB_FRAC = 0.10
assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC + CALIB_FRAC) - 1.0) < 1e-8

LEFT_FRAC = TEST_FRAC + CALIB_FRAC
CALIB_SHARE = CALIB_FRAC / LEFT_FRAC
TEST_SHARE  = TEST_FRAC  / LEFT_FRAC

# Helper functions:
def list_zip_batches(roots: List[Path]) -> List[Path]:
    zips = []
    for r in roots:
        if r.exists():
            zips.extend(sorted(r.glob("batch_*.zip")))
    assert zips, f"No batch_*.zip found under: {[str(r.resolve()) for r in roots]}"
    return zips

def build_index_by_speaker(zips: List[Path]) -> Dict[str, List[Dict]]:
    by_spk = {}
    for zp in zips:
        with zipfile.ZipFile(zp, "r") as zf:
            for name in zf.namelist():
                if not name.endswith(".pt"):
                    continue
                spk = name.split("/", 1)[0]
                by_spk.setdefault(spk, []).append({
                    "zip": zp.name,
                    "zip_dir": zp.parent.name,
                    "name": name,
                    "spk": spk,
                })
    return by_spk

def largest_remainder_counts(n_items: int, fractions: List[float]) -> List[int]:
    if n_items == 0:
        return [0]*len(fractions)
    targets = [f * n_items for f in fractions]
    floors  = [int(math.floor(t)) for t in targets]
    used    = sum(floors)
    rem     = n_items - used
    rema    = [t - fl for t, fl in zip(targets, floors)]
    order = sorted(range(len(fractions)), key=lambda i: rema[i], reverse=True)
    counts = floors[:]
    for i in range(rem):
        counts[order[i % len(fractions)]] += 1
    return counts

def make_base_train_val(by_spk: Dict[str, List[Dict]], seed: int):
    train, val, leftover = [], [], []
    rng = random.Random(seed)
    for spk, items in by_spk.items():
        items = items[:]
        rng.shuffle(items)
        n = len(items)
        c_train, c_val, c_left = largest_remainder_counts(n, [TRAIN_FRAC, VAL_FRAC, 1.0 - (TRAIN_FRAC + VAL_FRAC)])
        idx = 0
        if c_train > 0:
            train.extend(items[idx:idx+c_train]); idx += c_train
        if c_val > 0:
            val.extend(items[idx:idx+c_val]); idx += c_val
        if c_left > 0:
            leftover.extend(items[idx:idx+c_left])
    return train, val, leftover

def group_by_spk(items: List[Dict]) -> Dict[str, List[Dict]]:
    by = {}
    for x in items:
        by.setdefault(x["spk"], []).append(x)
    return by

def split_leftover_for_fold(leftover_by_spk: Dict[str, List[Dict]], seed: int):
    calib, test = [], []
    rng = random.Random(seed)
    for spk, items in leftover_by_spk.items():
        it = items[:]
        rng.shuffle(it)
        n = len(it)
        c_calib, c_test = largest_remainder_counts(n, [CALIB_SHARE, TEST_SHARE])
        idx = 0
        if c_calib > 0:
            calib.extend(it[idx:idx+c_calib]); idx += c_calib
        if c_test > 0:
            test.extend(it[idx:idx+c_test])
    return calib, test

def main():
    zips = list_zip_batches(ROOTS)
    print(f"Found {len(zips)} zip files under {ROOTS[0]}")

    by_spk = build_index_by_speaker(zips)
    speakers = sorted(by_spk.keys())
    print(f"Speakers: {len(speakers)}")

    train, val, leftover_all = make_base_train_val(by_spk, SEED_BASE)
    print("Base counts:", {"train": len(train), "val": len(val), "leftover": len(leftover_all)})
    leftover_by_spk = group_by_spk(leftover_all)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    fold_json_paths = []

    for k in range(K_FOLDS):
        calib, test = split_leftover_for_fold(leftover_by_spk, seed=SEED_BASE + 1000 + k)
        summary = {"train": len(train), "val": len(val), "calibration": len(calib), "test": len(test)}
        print(f"[fold {k}] {summary}")

        out = {
            "label_map": {spk: i for i, spk in enumerate(speakers)},
            "train": train,
            "val": val,
            "test": test,
            "calibration": calib,
            "split_info": {
                "seed_base": SEED_BASE,
                "fold_seed": SEED_BASE + 1000 + k,
                "fractions": {
                    "train": TRAIN_FRAC, "val": VAL_FRAC,
                    "test": TEST_FRAC, "calibration": CALIB_FRAC
                },
                "total_speakers": len(speakers),
                "zip_roots": [str(r) for r in ROOTS],
                "notes": "Train/val fixed; leftover split into calib/test differently per fold.",
            },
        }
        out_path = SPLITS_DIR / f"new_splits_fold{k}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        fold_json_paths.append(out_path)

    if fold_json_paths:
        (SPLITS_DIR / "new_splits.json").write_text(fold_json_paths[0].read_text(), encoding="utf-8")
        print(f"Wrote {SPLITS_DIR/'new_splits.json'} (alias of fold 0)")

    print("Wrote:", [str(p) for p in fold_json_paths])

if __name__ == "__main__":
    main()