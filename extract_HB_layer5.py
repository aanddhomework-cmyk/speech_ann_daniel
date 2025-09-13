import os
import io
import sys
import zipfile
from pathlib import Path
import traceback
import tqdm

import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor


vox_root_parent = Path(r"C:\Datasets\voxceleb1")
dataset_roots = [vox_root_parent / "wav", vox_root_parent / "test_wav"]

output_root = Path("embedding_batches_zip") 
output_root.mkdir(parents=True, exist_ok=True)

batch_size_files = 100         
target_sr = 16000
layer_idx = 5                  # HuBERT transformer layer 5
dtype_to_save = torch.float32  


log_dir = output_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
error_log_path = log_dir / "extract_errors.txt"
run_log_path   = log_dir / "run_log.txt"


temp_dir = output_root / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)

#Utilities functions:
def log_line(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

def pick_primary_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def clear_accel_cache():
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def list_all_wavs(roots):
    files = []
    for r in roots:
        if not r.exists():
            continue
        files.extend(sorted(r.rglob("*.wav")))
    return files

def index_existing_entries(out_root: Path) -> set:
    seen = set()
    for zp in sorted(out_root.glob("batch_*.zip")):
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                for n in zf.namelist():
                    seen.add(n)
        except zipfile.BadZipFile:
            log_line(error_log_path, f"[corrupt_zip] {zp}")
    return seen

def write_tensor_into_zip(zf: zipfile.ZipFile, arcname: str, tensor: torch.Tensor):
    buf = io.BytesIO()
    torch.save(tensor.to(dtype=dtype_to_save).cpu(), buf)
    info = zipfile.ZipInfo(arcname)
    info.compress_type = zipfile.ZIP_DEFLATED
    zf.writestr(info, buf.getvalue())

# Load model:
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True)
model.eval()

device = pick_primary_device()
model.to(device)
print(f"[device] Using: {device}")
log_line(run_log_path, f"[device] {device}")

wav_files = list_all_wavs(dataset_roots)
if not wav_files:
    msg = f"[fatal] No .wav files found under: {dataset_roots}"
    print(msg)
    log_line(error_log_path, msg)
    sys.exit(1)

print(f"[data] Found {len(wav_files)} WAV files across both folders.")
log_line(run_log_path, f"[data] total_wavs={len(wav_files)}")

already_in_zips = index_existing_entries(output_root)
print(f"[resume] Found {len(already_in_zips)} entries already inside existing zips.")
log_line(run_log_path, f"[resume] existing_entries={len(already_in_zips)}")

file_in_batch = 0
batch_count = 0
staged = [] 

def flush_zip_batch(staged_list, batch_idx):
    if not staged_list:
        return
    zip_path = output_root / f"batch_{batch_idx:03d}.zip"
    print(f"\n[zipping] batch {batch_idx} → {zip_path}  (files: {len(staged_list)})")
    log_line(run_log_path, f"[zipping] batch={batch_idx} files={len(staged_list)} -> {zip_path}")

    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        for full_path, arcname in staged_list:
            try:
                try:
                    zf.getinfo(arcname)
                    pass
                except KeyError:
                    if Path(full_path).exists():
                        zf.write(full_path, arcname=arcname)
                        already_in_zips.add(arcname)
                try:
                    os.remove(full_path)
                except Exception:
                    pass
            except Exception as e:
                err = f"[zip_write_error] arc={arcname} zip={zip_path} err={e}"
                print(err)
                log_line(error_log_path, err)

    staged_list.clear()

def relative_arcname(wav_path: Path) -> str:
    """
    Produce arcname relative to whichever root ('wav' or 'test_wav') it belongs to
    """
    for r in dataset_roots:
        try:
            rel = wav_path.relative_to(r)
            return str(rel.with_suffix(".pt")).replace("\\", "/")
        except ValueError:
            continue
    rel = wav_path.relative_to(wav_path.parents[1])
    return str(rel.with_suffix(".pt")).replace("\\", "/")

def load_wav_16k_mono(path: Path) -> torch.Tensor:
    wave, sr = torchaudio.load(str(path))
    if wave.dim() == 2 and wave.size(0) > 1:
        wave = wave.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wave = torchaudio.functional.resample(wave, orig_freq=sr, new_freq=target_sr)
    return wave.squeeze(0)  # (T,)

def forward_hidden_layer(wave_1d: torch.Tensor) -> torch.Tensor:
    """
    Returns full sequence embedding from chosen transformer layer (no pooling).
    Shape: (T_feat, 768)
    """
    assert 1 <= layer_idx <= model.config.num_hidden_layers, "layer_idx out of range (1..12)"
    inputs = extractor(wave_1d, sampling_rate=target_sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            out = model(**inputs)
            hs = out.hidden_states
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "device-side" in str(e).lower():
            warn = f"[warn] Accelerator error on file; retrying on CPU. err={e}"
            print(warn)
            log_line(run_log_path, warn)
            clear_accel_cache()
            model_cpu = model.to("cpu")
            inputs_cpu = {k: v.cpu() for k, v in inputs.items()}
            with torch.no_grad():
                out = model_cpu(**inputs_cpu)
                hs = out.hidden_states
            model.to(device)
        else:
            raise
    feats = hs[layer_idx].squeeze(0)
    return feats

progress = tqdm.tqdm(wav_files, desc=f"Extracting HuBERT L{layer_idx} (no pooling)")

try:
    for idx, wav_path in enumerate(progress):
        try:
            arcname = relative_arcname(wav_path)
            if arcname in already_in_zips:
                continue
            temp_pt = (temp_dir / arcname).resolve()
            temp_pt.parent.mkdir(parents=True, exist_ok=True)

            # Load audio and run model
            wave_1d = load_wav_16k_mono(wav_path)
            feats = forward_hidden_layer(wave_1d)

            # Save full sequence tensor (no mean pooling)
            torch.save(feats.to(dtype=dtype_to_save).cpu(), str(temp_pt))

            # Stage for zipping
            staged.append((str(temp_pt), arcname))
            file_in_batch += 1

            # Flush a zip when batch is full or at the very end
            if file_in_batch >= batch_size_files or idx == (len(wav_files) - 1):
                flush_zip_batch(staged, batch_count)
                batch_count += 1
                file_in_batch = 0

        except Exception as e_file:
            # Log per-file error & keep going
            tb = traceback.format_exc(limit=2)
            err = f"[file_error] {wav_path} :: {e_file}\n{tb}"
            print(err)
            log_line(error_log_path, err)

except KeyboardInterrupt:
    print("\n[interrupt] Stopping early by user request.")
    log_line(run_log_path, "[interrupt] User stopped the run.")
    flush_zip_batch(staged, batch_count)

except Exception as e_main:
    tb = traceback.format_exc()
    msg = f"[fatal] Unexpected error: {e_main}\n{tb}"
    print(msg)
    log_line(error_log_path, msg)
    flush_zip_batch(staged, batch_count)
    raise

print("\n[done] All possible files processed. Zips at:", output_root.resolve())
log_line(run_log_path, "[done] completed.")