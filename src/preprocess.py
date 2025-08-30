import os
import numpy as np
import mne
from scipy.signal import spectrogram

# --------------------------
# 1. Paths
# --------------------------
RAW_FOLDER = "data/physionet.org/files/hmc-sleep-staging/1.1/recordings"
PROCESSED_FOLDER = "data/processed"
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# --------------------------
# 2. Parameters
# --------------------------
SEGMENT_SEC = 30  # segment length in seconds
NFFT = 128        # spectrogram parameter
NOVERLAP = 64

# --------------------------
# 3. Helper Functions
# --------------------------
def to_spectrogram(signal, fs):
    """
    Convert 1D signal to log-scaled spectrogram
    """
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=NFFT, noverlap=NOVERLAP)
    Sxx = np.log(Sxx + 1e-10)
    return Sxx.astype(np.float32)  # shape: (freq_bins, time_steps)

def normalize_signal(signal):
    """
    Zero mean, unit variance
    """
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

# --------------------------
# 4. Main preprocessing
# --------------------------
edf_files = [
    f for f in os.listdir(RAW_FOLDER)
    if f.endswith('.edf') and not f.startswith('._') and "_sleepscoring" not in f
]
print(f"Found {len(edf_files)} EDF files.")

for file_name in edf_files:
    file_path = os.path.join(RAW_FOLDER, file_name)
    print(f"\nProcessing {file_name} ...")

    try:
        # Load EDF
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        if len(raw.ch_names) == 0:
            print(f"⚠️ Skipping {file_name} (no signal channels)")
            continue

        fs = int(raw.info['sfreq'])
        data = raw.get_data()  # shape: (channels, samples)
        n_channels, n_samples = data.shape

        segment_length = SEGMENT_SEC * fs
        n_segments = n_samples // segment_length

        for i in range(n_segments):
            start = i * segment_length
            end = start + segment_length
            segment = data[:, start:end]

            # Normalize per channel
            segment = np.array([normalize_signal(segment[ch]) for ch in range(n_channels)])

            # Convert each channel to spectrogram and stack
            specs = np.array([to_spectrogram(segment[ch], fs) for ch in range(n_channels)])

            # Dummy label (update later with apnea events)
            label = 0
            save_name = f"{file_name[:-4]}_seg{i}_label{label}.npy"
            save_path = os.path.join(PROCESSED_FOLDER, save_name)
            np.save(save_path, specs)

        print(f"✅ Saved {n_segments} segments for {file_name}")

    except Exception as e:
        print(f"❌ Skipping {file_name} due to error: {e}")

print("\n✅ Preprocessing complete! Spectrograms are saved in 'data/processed/'")