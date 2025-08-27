# @title data loader
# dataset_embeddings.py
# Loads last-layer HuBERT embeddings from ZIPs (created earlier),
# mean-pools to fixed length, and returns (x, y) samples for training.

from pathlib import Path
import io
import json
import zipfile
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class ZipEmbeddingDataset(Dataset):
    """
    Closed-set SID dataset that:
      - reads split spec from splits/closed_set_splits.json
      - streams .pt tensors from batch_*.zip without unzipping
      - mean-pools [T, 768] -> [768] and returns (x, y)

    Each item in the split has keys: {"zip": "batch_000.zip", "name": "id10001/.../00001.pt", "spk": "id10001"}
    """

    def __init__(
        self,
        root: str = "embedding_batches_zip",
        split_json: str = "splits/closed_set_splits.json",
        split: str = "train",
        normalize: bool = True,
        verbose: bool = True,
    ):
        self.root = Path(root)
        self.split_json = Path(split_json)
        self.split = split.lower()
        self.normalize = normalize
        self.verbose = verbose

        assert self.root.exists(), f"Embeddings folder not found: {self.root.resolve()}"
        assert self.split_json.exists(), f"Split file not found: {self.split_json.resolve()}"
        assert self.split in {"train", "val"}, f"split must be 'train' or 'val', got {self.split}"

        meta = json.load(open(self.split_json, "r"))
        self.label_map: Dict[str, int] = meta["label_map"]
        self.items: List[Dict[str, str]] = meta[self.split]

        # Cache of open ZipFile handles (reused across __getitem__ calls)
        self._zip_cache: Dict[str, zipfile.ZipFile] = {}

        if self.verbose:
            n_spk = len(self.label_map)
            print(
                f"[ZipEmbeddingDataset] split={self.split} | "
                f"speakers={n_spk} | items={len(self.items)} | root={self.root}"
            )

            # Show a few sample paths to confirm structure
            for i in range(min(3, len(self.items))):
                ex = self.items[i]
                print(f"  sample[{i}]: zip={ex['zip']} | name={ex['name']} | spk={ex['spk']} -> y={self.label_map[ex['spk']]}")

    def __len__(self) -> int:
        return len(self.items)

    def _get_zip(self, zip_name: str) -> zipfile.ZipFile:
        """Get (or open and cache) a ZipFile handle by filename."""
        zpath = str(self.root / zip_name)
        zf = self._zip_cache.get(zpath)
        if zf is None:
            # mode 'r' and allowZip64 True for large archives
            zf = zipfile.ZipFile(zpath, mode="r", allowZip64=True)
            self._zip_cache[zpath] = zf
        return zf

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        zf = self._get_zip(item["zip"])
        raw_bytes = zf.read(item["name"])  # bytes in memory
        # Load torch tensor directly from bytes
        x_seq: torch.Tensor = torch.load(io.BytesIO(raw_bytes), map_location="cpu")  # shape [T, 768]
        # Mean pooling over time -> [768]
        x = x_seq.mean(dim=0).float()
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=0)
        y = torch.tensor(self.label_map[item["spk"]], dtype=torch.long)
        return x, y

    def close(self):
        # Close any cached zip handles
        for zpath, zf in list(self._zip_cache.items()):
            try:
                zf.close()
            except Exception:
                pass
        self._zip_cache.clear()

    def __del__(self):
        # Ensure files are closed if the dataset gets GC'd
        try:
            self.close()
        except Exception:
            pass


# Quick sanity check / demo
if __name__ == "__main__":
    ds_train = ZipEmbeddingDataset(
        root="embedding_batches_zip",
        split_json="splits/closed_set_splits.json",
        split="train",
        normalize=True,
        verbose=True,
    )
    print(f"[demo] len(train) = {len(ds_train)}")

    # Inspect a couple of samples
    for i in range(2):
        x, y = ds_train[i]
        print(f"[demo] sample {i}: x.shape={tuple(x.shape)}  y={int(y)}")

    # Optional: build a DataLoader and fetch a batch
    # Use num_workers=0 on Windows if you see worker issues.
    dl = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    xb, yb = next(iter(dl))
    print(f"[demo] batch: xb.shape={tuple(xb.shape)}  yb.shape={tuple(yb.shape)}  yb[:8]={yb[:8].tolist()}")

    # Clean up handles
    ds_train.close()