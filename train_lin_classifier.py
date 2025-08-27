# @title training linear classifier

# train_linear.py
# Trains a linear classifier for closed-set SID on top of mean-pooled HuBERT embeddings (768-D)
# streamed directly from zipped batches. Uses Intel XPU when available; falls back to CPU per-batch if needed.
# This run will resume from BEST WEIGHTS if present (weights-only). We also SAVE optimizer/scheduler states
# so that future resumes can be fully seamless (but their absence will NOT block resuming this time).

from pathlib import Path
import json
import time
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm

from dataset_embeddings import ZipEmbeddingDataset  # expects mean-pooled 768-D vectors

# -----------------------
# Config
# -----------------------
SPLIT_JSON   = "splits/closed_set_splits.json"
EMB_ROOT     = "embedding_batches_zip"

BATCH_TRAIN  = 256
BATCH_VAL    = 512
EXTRA_EPOCHS = 30       # how many more epochs to run starting from the resumed epoch+1
LR           = 1e-3
WEIGHT_DECAY = 1e-4
RESUME       = True     # resume from best checkpoint if available (weights-only is fine)

NUM_WORKERS  = 0        # Windows-friendly; increase if stable
PIN_MEMORY   = True
PRINT_EVERY  = 50       # batches

FEATURE_DIM  = 768      # mean-pooled HuBERT-base last layer -> 768
CKPT_PATH    = "linear_sid_best.pt"

# -----------------------
# Device helpers
# -----------------------
def pick_primary_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

PRIMARY = pick_primary_device()
CPU = torch.device("cpu")

def to_device(t, device):
    return t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t

# -----------------------
# Train / Val loops with per-batch fallback
# -----------------------
def run_epoch(loader, model, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total, correct, loss_sum = 0, 0, 0.0
    running_loss, running_correct, running_total = 0.0, 0, 0

    pbar = tqdm.tqdm(loader, desc=("Train" if training else "Val"), file=sys.stdout)
    for i, (xb, yb) in enumerate(pbar, 1):
        try:
            xb = to_device(xb, PRIMARY)
            yb = to_device(yb, PRIMARY)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                loss.backward()
                optimizer.step()

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "device-side assert" in str(e).lower():
                tqdm.tqdm.write(f"[warn] Device error on batch {i}: {e}. Retrying on CPU for this batch.")
                try:
                    xb_cpu = to_device(xb, CPU)
                    yb_cpu = to_device(yb, CPU)
                    model_cpu = model.to(CPU)

                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model_cpu(xb_cpu)
                    loss = criterion(logits, yb_cpu)
                    if training:
                        loss.backward()
                        optimizer.step()

                    model = model.to(PRIMARY)  # back to primary for next batch

                except Exception as e2:
                    tqdm.tqdm.write(f"[error] CPU fallback failed on batch {i}: {e2}")
                    raise
            else:
                raise

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_sz = yb.size(0)
            batch_correct = (preds == yb).sum().item()
            total += batch_sz
            correct += batch_correct
            loss_sum += loss.item() * batch_sz

            running_total += batch_sz
            running_correct += batch_correct
            running_loss += loss.item() * batch_sz

        if (i % PRINT_EVERY) == 0:
            avg_loss = running_loss / max(1, running_total)
            avg_acc  = running_correct / max(1, running_total)
            tqdm.tqdm.write(f"[{('train' if training else 'val')}] step {i} "
                            f"avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f}")
            running_loss, running_correct, running_total = 0.0, 0, 0

    epoch_loss = loss_sum / max(1, total)
    epoch_acc  = correct / max(1, total)
    return epoch_loss, epoch_acc, model

# -----------------------
# Main
# -----------------------
def main():
    print(f"Primary device: {PRIMARY}")

    meta = json.load(open(SPLIT_JSON, "r"))
    num_classes = len(meta["label_map"])
    print(f"Classes (speakers): {num_classes}")

    train_ds = ZipEmbeddingDataset(root=EMB_ROOT, split_json=SPLIT_JSON, split="train",
                                   normalize=True, verbose=True)
    val_ds   = ZipEmbeddingDataset(root=EMB_ROOT, split_json=SPLIT_JSON, split="val",
                                   normalize=True, verbose=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_VAL, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = nn.Linear(FEATURE_DIM, num_classes).to(PRIMARY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # LR scheduler: reduce LR when val accuracy plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",       # stepping with validation accuracy
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )

    start_epoch = 1
    best_val_acc = 0.0

    # ---- Resume from best checkpoint (weights-only is fine) ----
    if RESUME and Path(CKPT_PATH).exists():
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        # Required: model weights
        model.load_state_dict(ckpt["state_dict"])
        # Metadata (for logging / deciding next epoch)
        best_val_acc = float(ckpt.get("meta", {}).get("best_val_acc", 0.0))
        prev_epochs  = int(ckpt.get("meta", {}).get("epochs_trained", 0))
        start_epoch  = prev_epochs + 1
        print(f"Resumed from {CKPT_PATH}: prev_epochs={prev_epochs}, best_val_acc={best_val_acc:.4f}")

        # Optional: if previous runs saved optimizer/scheduler, load them too
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("Loaded optimizer state from checkpoint.")
            except Exception as e:
                print(f"Could not load optimizer state (continuing fresh): {e}")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
                print("Loaded scheduler state from checkpoint.")
            except Exception as e:
                print(f"Could not load scheduler state (continuing fresh): {e}")
    else:
        print("No resume checkpoint found; training from scratch.")

    end_epoch = start_epoch + EXTRA_EPOCHS - 1

    start_time = time.time()
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\n=== Epoch {epoch}/{end_epoch} ===")
        tr_loss, tr_acc, model = run_epoch(train_loader, model, criterion, optimizer)
        va_loss, va_acc, model = run_epoch(val_loader,   model, criterion, optimizer=None)

        # Step scheduler on validation accuracy
        scheduler.step(va_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"lr {current_lr:.2e}")

        # Always save the best weights (and include optimizer/scheduler for future resumes)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "feat_dim": FEATURE_DIM,
                    "meta": {
                        "epochs_trained": epoch,
                        "best_val_acc": best_val_acc,
                    },
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                CKPT_PATH,
            )
            print(f"Saved best checkpoint: {CKPT_PATH} (val_acc={best_val_acc:.4f})")

    elapsed = time.time() - start_time
    print(f"\nDone. Best val acc: {best_val_acc:.4f}. Elapsed: {elapsed/60:.1f} min.")

    train_ds.close()
    val_ds.close()

if __name__ == "__main__":
    main()