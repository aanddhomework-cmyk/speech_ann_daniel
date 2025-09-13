from pathlib import Path
import io
import json
import zipfile
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class ZipEmbeddingDataset(Dataset):
    """
    Closed-set SID dataset
    """

    def __init__(
        self,
        roots: List[str] = ["embedding_batches_zip"],
        split_json: str = "splits/new_splits.json",
        split: str = "train",
        normalize: bool = True,
        verbose: bool = True,
    ):
        self.roots = {Path(r).name: Path(r) for r in roots}
        self.split_json = Path(split_json)
        self.split = split.lower()
        self.normalize = normalize
        self.verbose = verbose

        # Checks
        for r in self.roots.values():
            assert r.exists(), f"Embeddings folder not found: {r.resolve()}"
        assert self.split_json.exists(), f"Split file not found: {self.split_json.resolve()}"
        assert self.split in {"train", "val", "test", "calibration"}, \
            f"split must be one of train/val/test/calibration, got {self.split}"

        # Load split metadata
        meta = json.load(open(self.split_json, "r", encoding="utf-8"))
        self.label_map: Dict[str, int] = meta["label_map"]
        self.items: List[Dict[str, str]] = meta[self.split]

        # Cache of open ZipFile handles
        self._zip_cache: Dict[Tuple[str, str], zipfile.ZipFile] = {}

        if self.verbose:
            n_spk = len(self.label_map)
            print(
                f"[ZipEmbeddingDataset] split={self.split} | "
                f"speakers={n_spk} | items={len(self.items)} | roots={list(self.roots.keys())}"
            )
            if len(self.items) == 0:
                print("  [warn] No items in this split. Check your split JSON / fractions.")
            # Show a few sample paths to confirm structure
            for i in range(min(3, len(self.items))):
                ex = self.items[i]
                print(
                    f"  sample[{i}]: dir={ex['zip_dir']} | zip={ex['zip']} "
                    f"| name={ex['name']} | spk={ex['spk']} -> y={self.label_map[ex['spk']]}"
                )

            # Quick sanity: ensure all referenced roots are provided
            missing_roots = set(e["zip_dir"] for e in self.items) - set(self.roots.keys())
            if missing_roots:
                raise RuntimeError(
                    f"Split JSON references zip_dir(s) {sorted(missing_roots)} "
                    f"that are not in provided roots {list(self.roots.keys())}."
                )

    def __len__(self) -> int:
        return len(self.items)

    def _get_zip(self, zip_dir: str, zip_name: str) -> zipfile.ZipFile:
        """Get (or open and cache) a ZipFile handle by (zip_dir, zip_name)."""
        root = self.roots[zip_dir]
        zpath = str(root / zip_name)
        key = (zip_dir, zip_name)
        zf = self._zip_cache.get(key)
        if zf is None:
            zf = zipfile.ZipFile(zpath, mode="r", allowZip64=True)
            self._zip_cache[key] = zf
        return zf

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        zf = self._get_zip(item["zip_dir"], item["zip"])
        raw_bytes = zf.read(item["name"])
        x_seq: torch.Tensor = torch.load(io.BytesIO(raw_bytes), map_location="cpu")
        if x_seq.ndim != 2 or x_seq.shape[1] != 768:
            raise ValueError(f"Bad tensor shape for {item['name']}: expected [T,768], got {tuple(x_seq.shape)}")
        x = x_seq.mean(dim=0).float()
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=0)

        y = torch.tensor(self.label_map[item["spk"]], dtype=torch.long)
        return x, y

    def close(self):
        """Close any cached zip handles."""
        for _, zf in list(self._zip_cache.items()):
            try:
                zf.close()
            except Exception:
                pass
        self._zip_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

if __name__ == "__main__":
    split_json_path = "splits/new_splits.json"
    roots = ["embedding_batches_zip"]

    for split_name in ["train", "val", "calibration", "test"]:
        print(f"\n[demo] Opening split: {split_name}")
        ds = ZipEmbeddingDataset(
            roots=roots,
            split_json=split_json_path,
            split=split_name,
            normalize=True,
            verbose=True,
        )
        print(f"[demo] len({split_name}) = {len(ds)}")
        peek = min(2, len(ds))
        for i in range(peek):
            x, y = ds[i]
            print(f"[demo] sample {i}: x.shape={tuple(x.shape)}  y={int(y)}")
        ds.close()

    print("\n[demo] Building a DataLoader on train...")
    ds_train = ZipEmbeddingDataset(
        roots=roots,
        split_json=split_json_path,
        split="train",
        normalize=True,
        verbose=False,
    )
    dl = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    xb, yb = next(iter(dl))
    print(f"[demo] batch: xb.shape={tuple(xb.shape)}  yb.shape={tuple(yb.shape)}")
    ds_train.close()