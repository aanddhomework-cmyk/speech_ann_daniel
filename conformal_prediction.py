from mean_pool_data_loader import ZipEmbeddingDataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import torch

class ConformalPredictor:
    def __init__(self, alpha=0.1, method="APS", raps_reg=1.0, raps_kreg=5):
        """
        alpha: miscoverage rate (1 - desired coverage)
        Scoring method: 'Basic', 'APS', 'RAPS'
        raps_reg: lambda (ℓ2 regularization strength) for RAPS
        raps_kreg: k (slack parameter) for RAPS
        """
        self.alpha = alpha
        self.method = method
        self.qhat = None
        self.raps_reg = raps_reg
        self.raps_kreg = raps_kreg

    def _get_scores(self, smx, labels=None):
        """
        Return conformity scores depending on method.
        If labels is provided, return scores for the true labels.
        """
        n, C = smx.shape
        pi = smx.argsort(1)[:, ::-1]              
        srt = np.take_along_axis(smx, pi, axis=1) 

        if self.method == "Basic":
            # Conformity score = 1 - prob(true_label)
            if labels is None:
                raise ValueError("labels required for Basic fit.")
            scores = 1 - smx[np.arange(n), labels]

        elif self.method == "APS":
            # Conformity score = cumulative sum up to true label
            cumsum = srt.cumsum(axis=1)
            inv_pi = pi.argsort(axis=1)
            scores = cumsum[np.arange(n), inv_pi[np.arange(n), labels]]

        elif self.method == "RAPS":
            # RAPS = APS with randomization + penalties
            cumsum = srt.cumsum(axis=1)
            inv_pi = pi.argsort(axis=1)
            ranks = inv_pi[np.arange(n), labels]
            # penalty = λ * max(0, rank - kreg)
            penalties = self.raps_reg * np.maximum(0, ranks - self.raps_kreg)
            scores = cumsum[np.arange(n), ranks] + penalties
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return scores

    def fit(self, cal_smx, cal_labels):
        """Fit conformal predictor using calibration softmax + labels."""
        scores = self._get_scores(cal_smx, cal_labels)
        n = len(scores)
        self.qhat = np.quantile(
            scores,
            np.ceil((n + 1) * (1 - self.alpha)) / n,
            method="higher"
        )

    def predict(self, smx):
        """Return prediction sets under chosen method."""
        if self.qhat is None:
            raise ValueError("Call fit() first.")
        n, C = smx.shape
        pi = smx.argsort(1)[:, ::-1]
        srt = np.take_along_axis(smx, pi, axis=1)

        if self.method == "Basic":
            # Include all classes with 1 - p <= qhat  <=>  p >= 1 - qhat
            pred_mask = srt >= 1 - self.qhat

        elif self.method == "APS":
            cumsum = srt.cumsum(axis=1)
            pred_mask = cumsum <= self.qhat

        elif self.method == "RAPS":
            cumsum = srt.cumsum(axis=1)
            ranks = np.arange(C)[None, :]
            penalties = self.raps_reg * np.maximum(0, ranks - self.raps_kreg)
            adjusted = cumsum + penalties
            pred_mask = adjusted <= self.qhat

        else:
            raise ValueError(f"Unknown method: {self.method}")

        sets = [pi[i][pred_mask[i]] for i in range(n)]
        return sets

# Device functions:
def pick_primary_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(t, device):
    return t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t

def filter_existing_roots(roots):
    exist = [str(r) for r in roots if Path(r).exists()]
    miss  = [str(r) for r in roots if not Path(r).exists()]
    if miss:
        print(f"[warn] Ignoring missing roots: {miss}")
    if not exist:
        raise FileNotFoundError("No valid embedding roots found.")
    return exist


#Helper functions:
@torch.no_grad()
def load_linear_head(ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = int(ckpt.get("num_classes"))
    feat_dim    = int(ckpt.get("feat_dim", 768))
    model = torch.nn.Linear(feat_dim, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(PRIMARY)
    print(f"[model] loaded {ckpt_path} | feat_dim={feat_dim} num_classes={num_classes}")
    return model

def collect_softmax_and_labels(split_name: str, model):
    roots = filter_existing_roots(EMB_ROOTS)
    ds = ZipEmbeddingDataset(
        roots=roots, split_json=SPLIT_JSON, split=split_name,
        normalize=True, verbose=True
    )
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            try:
                xb = to_device(xb, PRIMARY).float()
                logits = model(xb)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "device-side assert" in str(e).lower():
                    print("[warn] Accelerator hiccup; retrying this batch on CPU.")
                    logits = model.to(CPU)(xb.to(CPU))
                    model.to(PRIMARY)
                else:
                    raise
            smx = torch.softmax(logits, dim=1).to(CPU).numpy()
            probs.append(smx)
            labels.append(yb.to(CPU).numpy().astype(np.int64))

    P = np.concatenate(probs, axis=0) if probs else np.zeros((0, model.out_features), dtype=np.float32)
    y = np.concatenate(labels, axis=0) if labels else np.zeros((0,), dtype=np.int64)
    ds.close()
    print(f"[data] {split_name}: softmax={P.shape} labels={y.shape}")
    return P, y

def summarize_probs(name, P, y):
    if P.size == 0:
        print(f"[sanity:{name}] EMPTY")
        return
    pmax = P.max(axis=1)
    top1 = (P.argmax(axis=1) == y).mean() if y.size else float("nan")
    print(f"[sanity:{name}] top1={top1:.4f}  pmax mean={pmax.mean():.4f}  median={np.median(pmax):.4f}  "
          f"q10={np.quantile(pmax,0.10):.4f}  q90={np.quantile(pmax,0.90):.4f}")
    print(f"[sanity:{name}] sum(P_i) mean={P.sum(1).mean():.6f} (should be ~1.0)")

def run_conformal_on_split(split_json, score_method):
    global SPLIT_JSON
    SPLIT_JSON = split_json
    print(f"[device] PRIMARY: {PRIMARY}")
    print(f"[cfg] split_json={SPLIT_JSON} | roots={EMB_ROOTS} | alpha={ALPHA} | score={score_method}")

    # 1) Load linear head
    model = load_linear_head(CKPT_PATH)

    # 2) Collect softmax
    cal_P, cal_y = collect_softmax_and_labels("calibration", model)
    test_P, test_y = collect_softmax_and_labels("test", model)

    # 3) Sanity
    summarize_probs("calib", cal_P, cal_y)
    summarize_probs("test",  test_P, test_y)

    # 4) Fit conformal
    print(f"[cp] Fitting ConformalPredictor (alpha={ALPHA}, method={score_method})...")
    cp = ConformalPredictor(alpha=ALPHA, method=score_method,
                            raps_reg=RAPS_REG, raps_kreg=RAPS_KREG)
    cp.fit(cal_P, cal_y)
    print(f"[cp] qhat = {cp.qhat:.6f}")

    # 5) Predict sets
    print("[cp] Predicting sets on test...")
    sets = cp.predict(test_P)
    set_sizes = np.fromiter((len(s) for s in sets), dtype=np.int32, count=len(sets))

    # 6) Metrics
    contained = np.fromiter((test_y[i] in sets[i] for i in range(len(test_y))), dtype=bool, count=len(test_y))
    coverage = contained.mean() if contained.size else float("nan")
    top1 = (test_P.argmax(axis=1) == test_y).mean() if test_P.size else float("nan")
    avg_set_size = set_sizes.mean()
    median_set_size = np.median(set_sizes)
    q90_set_size = np.quantile(set_sizes,0.90)
    max_set_size = set_sizes.max() if set_sizes.size else 0

    print(f"[metrics] coverage={coverage:.4f}  avg_set_size={avg_set_size:.2f}  "
          f"median_set={median_set_size:.1f}  q90_set={q90_set_size:.1f}  "
          f"max_set={max_set_size}  top1={top1:.4f}")

    return {
        "coverage": coverage,
        "avg_set_size": avg_set_size,
        "median_set_size": median_set_size,
        "q90_set_size": q90_set_size,
        "max_set_size": max_set_size,
        "top1_acc": top1
    }

if __name__ == "__main__":
    #Devices:
    PRIMARY = pick_primary_device()
    CPU = torch.device("cpu")

    #Config:
    EMB_ROOTS    = ["embedding_batches_zip"]
    CKPT_PATH    = "linear_sid_best.pt"
    BATCH        = 512
    NUM_WORKERS  = 0
    PIN_MEMORY   = True
    ALPHA        = 0.10

    SPLIT_FILES = [
    "splits/new_splits_fold0.json",
    "splits/new_splits_fold1.json",
    "splits/new_splits_fold2.json",
    "splits/new_splits_fold3.json",
    "splits/new_splits_fold4.json",
    ]

    # RAPS parameters:
    RAPS_REG     = 1.0
    RAPS_KREG    = 5

    
    SCORE = "RAPS"  # "RAPS" or "Basic" or "APS"
    all_metrics = []
    for split_file in SPLIT_FILES:
        print(f"\n===== Running on split: {split_file} =====")
        metrics = run_conformal_on_split(split_file, SCORE)
        all_metrics.append(metrics)

    # Aggregate results
    keys = all_metrics[0].keys()
    print("\n==== Averaged Results across 5 splits ====")
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        print(f"{k}: {vals.mean():.4f} ± {vals.std():.4f}")

    print("\nMethod\tcoverage\tavg_set_size\tmedian_set_size\tq90_set_size\tmax_set_size\ttop1_acc")
    means = {k: np.mean([m[k] for m in all_metrics]) for k in keys}
    stds = {k: np.std([m[k] for m in all_metrics]) for k in keys}
    print(f"{SCORE}\t{means['coverage']:.4f}±{stds['coverage']:.4f}\t"
          f"{means['avg_set_size']:.2f}±{stds['avg_set_size']:.2f}\t"
          f"{means['median_set_size']:.1f}±{stds['median_set_size']:.1f}\t"
          f"{means['q90_set_size']:.1f}±{stds['q90_set_size']:.1f}\t"
          f"{means['max_set_size']:.0f}±{stds['max_set_size']:.0f}\t"
          f"{means['top1_acc']:.4f}±{stds['top1_acc']:.4f}")