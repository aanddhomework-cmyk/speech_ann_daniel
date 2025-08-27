# @title plot curves
# plot_training_curves.py  (robust, two-line tolerant)

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)
TRAIN_RE = re.compile(r"train\s*loss[:=]?\s*([0-9.]+).*?acc[:=]?\s*([0-9.]+)", re.IGNORECASE)
VAL_RE   = re.compile(r"val\s*loss[:=]?\s*([0-9.]+).*?acc[:=]?\s*([0-9.]+)",   re.IGNORECASE)

def try_parse(line):
    """Return (epoch, tr_loss, tr_acc, va_loss, va_acc) or None."""
    me = EPOCH_RE.search(line)
    if not me:
        return None
    mt = TRAIN_RE.search(line)
    mv = VAL_RE.search(line)
    if mt and mv:
        return (int(me.group(1)),
                float(mt.group(1)), float(mt.group(2)),
                float(mv.group(1)), float(mv.group(2)))
    return (int(me.group(1)), None, None, None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logfile", default="run_log_classifier.txt")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    log_path = Path(args.logfile)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    epochs, tr_loss, tr_acc, va_loss, va_acc = [], [], [], [], []
    matched_prints = []
    candidates = misses_same = parsed_next = 0

    i = 0
    while i < len(lines):
        raw = lines[i]
        if "Epoch" not in raw:
            i += 1
            continue

        # First, try same line
        parsed = try_parse(raw)
        if parsed and parsed[1] is not None:
            e, tl, ta, vl, va = parsed
            epochs.append(e); tr_loss.append(tl); tr_acc.append(ta); va_loss.append(vl); va_acc.append(va)
            matched_prints.append(f"[same] {raw.strip()}")
        else:
            # If epoch found but metrics missing, try concatenating the next line
            candidates += 1
            nxt = lines[i+1] if i+1 < len(lines) else ""
            both = (raw + "  " + nxt)
            parsed2 = try_parse(both)
            if parsed2 and parsed2[1] is not None:
                e, tl, ta, vl, va = parsed2
                epochs.append(e); tr_loss.append(tl); tr_acc.append(ta); va_loss.append(vl); va_acc.append(va)
                matched_prints.append(f"[next] {raw.strip()}  ||  {nxt.strip()}")
                parsed_next += 1
                i += 1  # we consumed the next line too
            else:
                misses_same += 1
        i += 1

    if not epochs:
        print(f"No epoch summaries parsed from {log_path}.")
        print("Showing first 5 lines that contain 'Epoch':")
        shown = 0
        for L in lines:
            if "Epoch" in L:
                print("  ", L.strip())
                shown += 1
                if shown == 5:
                    break
        return

    # Sort by epoch, dedupe (keep last by epoch if duplicated)
    bundle = {}
    for e, tl, ta, vl, va in zip(epochs, tr_loss, tr_acc, va_loss, va_acc):
        bundle[e] = (tl, ta, vl, va)
    epochs = sorted(bundle.keys())
    tr_loss = [bundle[e][0] for e in epochs]
    tr_acc  = [bundle[e][1] for e in epochs]
    va_loss = [bundle[e][2] for e in epochs]
    va_acc  = [bundle[e][3] for e in epochs]

    print(f"Parsed {len(epochs)} epochs from {log_path}.")
    print(f"Matched on same line: {len(matched_prints) - parsed_next}, via next line: {parsed_next}, misses: {misses_same}.")
    for s in matched_prints[:3]:
        print("  ", s)

    # Plot loss
    import matplotlib
    plt.figure()
    plt.plot(epochs, tr_loss, label="train loss")
    plt.plot(epochs, va_loss, label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs epoch")
    plt.grid(True, alpha=0.3); plt.legend()
    lp = outdir / "loss.png"; plt.savefig(lp, bbox_inches="tight"); print(f"Saved {lp}")

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, tr_acc, label="train acc")
    plt.plot(epochs, va_acc, label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Accuracy vs epoch")
    plt.grid(True, alpha=0.3); plt.legend()
    ap = outdir / "accuracy.png"; plt.savefig(ap, bbox_inches="tight"); print(f"Saved {ap}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()