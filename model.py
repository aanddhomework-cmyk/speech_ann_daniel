# @title using frozen hubert base model to extract the layer of embeddings for voxceleb 1 dataset
import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from pathlib import Path
import zipfile
import tqdm
import os

# === Settings ===
dataset_root = Path(r"C:\Datasets\voxceleb1\wav")
output_root = Path("embedding_batches_zip")
output_root.mkdir(parents=True, exist_ok=True)
batch_size = 100
temp_dir = output_root / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)

# === Load model & feature extractor ===
extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
model = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True)
model.eval()

# === Device selection ===
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
model.to(device)
print(f"Model running on device: {device}")

# === List all .wav files ===
wav_files = list(dataset_root.rglob("*.wav"))
print(f"Found {len(wav_files)} WAV files to process.")

# === Processing ===
batch_count = 0
file_in_batch = 0
pt_paths = []

for idx, wav_path in enumerate(tqdm.tqdm(wav_files, desc="Extracting embeddings")):
    try:
        # Output paths
        rel_path = wav_path.relative_to(dataset_root)
        pt_path = temp_dir / rel_path.with_suffix(".pt")
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        pt_paths.append((pt_path, rel_path.with_suffix(".pt")))

        # Load and resample
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)

        inputs = extractor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run model
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.hidden_states
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.xpu.is_available():
                    torch.xpu.empty_cache()
                else:
                    torch.cuda.empty_cache()
                print(f"OOM on device at {wav_path}, retrying on CPU")
                model_cpu = model.to("cpu")
                inputs = {k: v.cpu() for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model_cpu(**inputs)
                    hidden_states = outputs.hidden_states
                model.to(device)
            else:
                raise e

        # === Only save the LAST layer ===
        last_embedding = hidden_states[-1].squeeze(0).cpu()
        torch.save(last_embedding, pt_path)
        file_in_batch += 1

        # === Compress batch if full ===
        if file_in_batch == batch_size or idx == len(wav_files) - 1:
            zip_path = output_root / f"batch_{batch_count:03d}.zip"
            print(f"\nZipping batch {batch_count} to {zip_path}")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for full_path, arc_path in pt_paths:
                    if full_path.exists():
                        zipf.write(full_path, arcname=str(arc_path))
            # Delete files
            for full_path, _ in pt_paths:
                try:
                    os.remove(full_path)
                except Exception as e:
                    print(f"Failed to delete {full_path}: {e}")
            pt_paths = []
            file_in_batch = 0
            batch_count += 1

    except Exception as e:
        print(f"Error processing {wav_path}: {e}")