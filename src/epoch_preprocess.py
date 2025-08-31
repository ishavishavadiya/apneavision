# apneavision/src/epoch_preprocess.py
import os
import json
import numpy as np
from numpy.lib.format import open_memmap

# ---- Paths ----
PROCESSED_DIR = "apneavision/data/processed"
OUTPUT_DIR = "apneavision/data/epochs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Constants (adjust if needed) ----
SFREQ = 200           # Hz
EPOCH_LENGTH = 30     # seconds
SAMPLES_PER_EPOCH = SFREQ * EPOCH_LENGTH  # 6000 samples
CHUNK_EPOCHS = 128    # how many epochs to process at once per subject (reduce if memory is tight)


def subject_epoch_info(sn_id: str):
    """
    Return (n_channels, n_epochs_valid) for a subject WITHOUT loading large arrays into RAM.
    n_epochs_valid accounts for labels != -1 and signal length.
    """
    sig_file = os.path.join(PROCESSED_DIR, f"{sn_id}_signals.npy")
    lab_file = os.path.join(PROCESSED_DIR, f"{sn_id}_labels.npy")

    if not (os.path.exists(sig_file) and os.path.exists(lab_file)):
        return None

    labels = np.load(lab_file, mmap_mode="r")
    # keep only labeled (exclude -1)
    valid_mask = labels != -1
    n_labeled = int(np.count_nonzero(valid_mask))

    sigs = np.load(sig_file, mmap_mode="r")  # shape (C, T)
    n_channels = int(sigs.shape[0])
    n_possible = int(sigs.shape[1] // SAMPLES_PER_EPOCH)
    n_epochs = int(min(n_labeled, n_possible))

    if n_epochs <= 0:
        return None

    return n_channels, n_epochs


def iter_subject_epoch_chunks(sn_id: str, n_epochs: int, n_channels: int):
    """
    Yield (epochs_chunk, labels_chunk) for a subject in small batches.
    epochs_chunk shape: (batch_E, C, SAMPLES_PER_EPOCH), dtype float32
    labels_chunk shape: (batch_E,), dtype like original (we’ll cast to int16)
    """
    sig_file = os.path.join(PROCESSED_DIR, f"{sn_id}_signals.npy")
    lab_file = os.path.join(PROCESSED_DIR, f"{sn_id}_labels.npy")

    signals = np.load(sig_file, mmap_mode="r")   # (C, T)
    labels = np.load(lab_file, mmap_mode="r")    # (N_all,)

    # Filter labels != -1 and truncate to n_epochs
    valid_mask = labels != -1
    labels = labels[valid_mask][:n_epochs]

    # Process in small chunks
    for start_e in range(0, n_epochs, CHUNK_EPOCHS):
        end_e = min(n_epochs, start_e + CHUNK_EPOCHS)
        ne = end_e - start_e
        start_samp = start_e * SAMPLES_PER_EPOCH
        end_samp = end_e * SAMPLES_PER_EPOCH

        # Slice the continuous signal and reshape to epochs for this chunk
        # signals[:, start_samp:end_samp] -> (C, ne*SPP)
        chunk = signals[:, start_samp:end_samp]
        # Reshape to (C, ne, SPP) -> (ne, C, SPP)
        chunk = chunk.reshape(n_channels, ne, SAMPLES_PER_EPOCH).transpose(1, 0, 2)
        # Downcast to float32 to save space
        chunk = chunk.astype(np.float32, copy=False)

        yield chunk, labels[start_e:end_e]


def main():
    # -------- Pass 0: discover subjects --------
    subjects = sorted([
        f.replace("_signals.npy", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith("_signals.npy")
        and os.path.exists(os.path.join(PROCESSED_DIR, f.replace("_signals.npy", "_labels.npy")))
    ])

    if not subjects:
        print("❌ No subjects found in", PROCESSED_DIR)
        return

    # -------- Pass 1: count total epochs and determine channels --------
    per_subject_info = []
    total_epochs = 0
    n_channels_ref = None

    print("Scanning subjects (pass 1/2) ...")
    for sn_id in subjects:
        info = subject_epoch_info(sn_id)
        if info is None:
            print(f"Skipping {sn_id}: no usable epochs.")
            continue
        n_channels, n_epochs = info

        if n_channels_ref is None:
            n_channels_ref = n_channels
        elif n_channels != n_channels_ref:
            # If any subject has a different channel count, skip it to keep shapes consistent
            print(f"⚠️ Skipping {sn_id}: channel mismatch ({n_channels} vs expected {n_channels_ref}).")
            continue

        per_subject_info.append((sn_id, n_epochs))
        total_epochs += n_epochs

    if total_epochs == 0 or n_channels_ref is None:
        print("❌ No usable data after scanning subjects.")
        return

    print(f"✅ Will write {total_epochs} epochs with {n_channels_ref} channels of {SAMPLES_PER_EPOCH} samples each.")

    # -------- Prepare memory-mapped .npy outputs --------
    epochs_path = os.path.join(OUTPUT_DIR, "epochs.npy")
    labels_path = os.path.join(OUTPUT_DIR, "labels.npy")

    # Create .npy files that we can write into incrementally
    epochs_mm = open_memmap(
        epochs_path, mode="w+",
        dtype=np.float32,
        shape=(total_epochs, n_channels_ref, SAMPLES_PER_EPOCH)
    )
    labels_mm = open_memmap(
        labels_path, mode="w+",
        dtype=np.int16,  # enough for 0..4
        shape=(total_epochs,)
    )

    # Save meta for convenience
    meta = {
        "epochs_path": epochs_path,
        "labels_path": labels_path,
        "shape": [total_epochs, n_channels_ref, SAMPLES_PER_EPOCH],
        "dtype_epochs": "float32",
        "dtype_labels": "int16",
        "sfreq": SFREQ,
        "epoch_length_sec": EPOCH_LENGTH,
        "samples_per_epoch": SAMPLES_PER_EPOCH,
        "chunk_epochs": CHUNK_EPOCHS,
        "subjects": [{"id": sn, "n_epochs": int(ne)} for sn, ne in per_subject_info],
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # -------- Pass 2: write data in chunks --------
    print("Writing epochs (pass 2/2) ...")
    cursor = 0
    for sn_id, n_epochs in per_subject_info:
        print(f"Processing {sn_id} ...")
        for chunk_X, chunk_y in iter_subject_epoch_chunks(sn_id, n_epochs, n_channels_ref):
            ne = chunk_X.shape[0]
            epochs_mm[cursor:cursor + ne] = chunk_X
            labels_mm[cursor:cursor + ne] = chunk_y.astype(np.int16, copy=False)
            cursor += ne

    # Ensure data is flushed
    del epochs_mm
    del labels_mm

    # -------- Quick sanity stats (using memmap read, not full RAM) --------
    labels = np.load(labels_path, mmap_mode="r")
    uniq, cnt = np.unique(labels, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    print(f"✅ Final dataset written to {OUTPUT_DIR}")
    print(f"   epochs.npy shape: {tuple(meta['shape'])}, dtype: {meta['dtype_epochs']}")
    print(f"   labels.npy shape: ({meta['shape'][0]},), dtype: {meta['dtype_labels']}")
    print("📊 Label distribution:", dist)


if __name__ == "__main__":
    main()
