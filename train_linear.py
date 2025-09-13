from mean_pool_data_loader import ZipEmbeddingDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from torch import nn
import json
import time
import sys
import ast
import torch
import tqdm

#config:
SPLIT_JSON   = "splits/new_splits.json"
EMB_ROOT     = ["embedding_batches_zip"]

BATCH_TRAIN  = 256
BATCH_VAL    = 512
EXTRA_EPOCHS = 30 
LR           = 1e-3
WEIGHT_DECAY = 1e-4
RESUME       = True

NUM_WORKERS  = 0  
PIN_MEMORY   = True
PRINT_EVERY  = 50 

FEATURE_DIM  = 76
CKPT_PATH    = "linear_sid_best.pt"

PLOTS_DIR    = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
HIST_JSON    = PLOTS_DIR / "history.json"
LEGACY_TXT   = PLOTS_DIR / "plot_log.txt"

APPEND_PLOTS_ON_RESUME = True

#device functions:
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

#plot:
def load_history():
    """Load history from JSON."""
    if HIST_JSON.exists():
        try:
            with open(HIST_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    if LEGACY_TXT.exists():
        try:
            txt = LEGACY_TXT.read_text(encoding="utf-8")
            hist = {}
            for line in txt.strip().splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        hist[k] = ast.literal_eval(v)
                    except Exception:
                        hist[k] = v
            out = {}
            if "epochs" in hist:
                out["epochs"] = hist["epochs"]
            if "train_loss" in hist:
                out["tr_losses"] = hist["train_loss"]
            if "val_loss" in hist:
                out["va_losses"] = hist["val_loss"]
            if "train_acc" in hist:
                out["tr_accs"] = hist["train_acc"]
            if "val_acc" in hist:
                out["va_accs"] = hist["val_acc"]
            return out
        except Exception:
            pass
    return None

def save_history(epochs, tr_losses, va_losses, tr_accs, va_accs, best_val_acc):
    hist = {
        "epochs": epochs,
        "tr_losses": tr_losses,
        "va_losses": va_losses,
        "tr_accs": tr_accs,
        "va_accs": va_accs,
        "best_val_acc": best_val_acc,
    }
    with open(HIST_JSON, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

#train/validation func:
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
                    model = model.to(PRIMARY)
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

def main():
    print(f"Primary device: {PRIMARY}")

    meta = json.load(open(SPLIT_JSON, "r", encoding="utf-8"))
    num_classes = len(meta["label_map"])
    print(f"Classes (speakers): {num_classes}")

    train_ds = ZipEmbeddingDataset(
        roots=EMB_ROOT, split_json=SPLIT_JSON, split="train",
        normalize=True, verbose=True
    )
    val_ds   = ZipEmbeddingDataset(
        roots=EMB_ROOT, split_json=SPLIT_JSON, split="val",
        normalize=True, verbose=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_TRAIN, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader   = DataLoader(
        val_ds, batch_size=BATCH_VAL, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = nn.Linear(FEATURE_DIM, num_classes).to(PRIMARY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-6,
    )

    start_epoch = 1
    best_val_acc = 0.0

    epochs_hist, tr_losses, va_losses, tr_accs, va_accs = [], [], [], [], []

    resumed = False
    prev_epochs = 0
    if RESUME and Path(CKPT_PATH).exists():
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        best_val_acc = float(ckpt.get("meta", {}).get("best_val_acc", 0.0))
        prev_epochs  = int(ckpt.get("meta", {}).get("epochs_trained", 0))
        start_epoch  = prev_epochs + 1
        resumed = True
        print(f"Resumed from {CKPT_PATH}: prev_epochs={prev_epochs}, best_val_acc={best_val_acc:.4f}")

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


    if resumed and APPEND_PLOTS_ON_RESUME:
        old = load_history()
        if old:
            try:
                old_epochs = old.get("epochs", [])
                if old_epochs and old_epochs[-1] == prev_epochs:
                    epochs_hist = old_epochs[:]       # copy
                    tr_losses   = old.get("tr_losses", [])[:]
                    va_losses   = old.get("va_losses", [])[:]
                    tr_accs     = old.get("tr_accs", [])[:]
                    va_accs     = old.get("va_accs", [])[:]
                    print(f"[plots] Loaded previous history up to epoch {prev_epochs} from {HIST_JSON if HIST_JSON.exists() else LEGACY_TXT}")
                else:
                    print("[plots] Found old history but epoch alignment did not match; starting fresh history for this run.")
            except Exception as e:
                print(f"[plots] Failed to load/merge old history: {e}")

    end_epoch = start_epoch + EXTRA_EPOCHS - 1

    start_time = time.time()
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\n=== Epoch {epoch}/{end_epoch} ===")
        tr_loss, tr_acc, model = run_epoch(train_loader, model, criterion, optimizer)
        va_loss, va_acc, model = run_epoch(val_loader,   model, criterion, optimizer=None)

        epochs_hist.append(epoch)
        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        va_losses.append(va_loss); va_accs.append(va_acc)

        scheduler.step(va_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"lr {current_lr:.2e}")

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

        save_history(epochs_hist, tr_losses, va_losses, tr_accs, va_accs, best_val_acc)

    elapsed = time.time() - start_time
    print(f"\nDone. Best val acc: {best_val_acc:.4f}. Elapsed: {elapsed/60:.1f} min.")
    try:
        # Loss
        plt.figure()
        plt.plot(epochs_hist, tr_losses, label="train loss")
        plt.plot(epochs_hist, va_losses, label="val loss")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training/Validation Loss")
        plt.legend(); plt.tight_layout()
        loss_path = PLOTS_DIR / "loss.png"
        plt.savefig(loss_path); plt.close()
        print(f"[plot] Saved {loss_path}")

        # Accuracy
        plt.figure()
        plt.plot(epochs_hist, tr_accs, label="train acc")
        plt.plot(epochs_hist, va_accs, label="val acc")
        plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Training/Validation Accuracy")
        plt.legend(); plt.tight_layout()
        acc_path = PLOTS_DIR / "accuracy.png"
        plt.savefig(acc_path); plt.close()
        print(f"[plot] Saved {acc_path}")
        with open(LEGACY_TXT, "w", encoding="utf-8") as f:
            f.write(f"best_val_acc={best_val_acc:.6f}\n")
            f.write(f"epochs={epochs_hist}\n")
            f.write(f"train_loss={tr_losses}\nval_loss={va_losses}\n")
            f.write(f"train_acc={tr_accs}\nval_acc={va_accs}\n")
        print(f"[plot] Wrote {LEGACY_TXT}")
    except Exception as e:
        print(f"[plot] Failed to save plots: {e}")

    # Cleanup
    train_ds.close()
    val_ds.close()

if __name__ == "__main__":
    main()