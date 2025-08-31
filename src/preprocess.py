# preprocess.py
import os
import mne
import numpy as np

# Paths
BASE_DIR = "apneavision/data/physionet.org/files/hmc-sleep-staging/1.1/recordings"
OUTPUT_DIR = "apneavision/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sleep stage mapping (merging N3 + N4)
# Sleep stage mapping
STAGE_MAPPING = {
    "Sleep stage W": 0,   # Wake
    "Sleep stage N1": 1,  # N1
    "Sleep stage N2": 2,  # N2
    "Sleep stage N3": 3,  # N3 (already merged N3+N4 in this dataset)
    "Sleep stage R": 4,   # REM
    # For datasets that might still have "1","2","3","4" style:
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
}


def preprocess_subject(sn_id: str):
    """Load one subject's EDF + scoring and save numpy arrays."""
    edf_file = os.path.join(BASE_DIR, f"{sn_id}.edf")
    annot_file = os.path.join(BASE_DIR, f"{sn_id}_sleepscoring.edf")

    if not os.path.exists(edf_file) or not os.path.exists(annot_file):
        print(f"⚠️ Skipping {sn_id}: missing files")
        return

    print(f"Processing {sn_id} ...")

    # Load PSG recording
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

    # Load scoring annotations
    annotations = mne.read_annotations(annot_file)
    raw.set_annotations(annotations)

    # Convert annotations into event indices
    events, event_ids = mne.events_from_annotations(raw)

    used_desc = sorted(set([ann["description"] for ann in annotations]))
    print(f"Used Annotations descriptions: {used_desc}")

    # Convert annotations into labels
    labels = []
    for ann in annotations:
        desc = ann["description"]
        if desc in STAGE_MAPPING:
            labels.append(STAGE_MAPPING[desc])
        else:
            labels.append(-1)  # unknown / lights on/off / ?
    labels = np.array(labels)

    # Extract EEG channels (first EEG channel for simplicity, can expand)
    picks = mne.pick_types(raw.info, eeg=True, eog=False, emg=False, ecg=False)
    data, _ = raw[picks, :]  # shape (n_channels, n_times)

    # Save subject data
    np.save(os.path.join(OUTPUT_DIR, f"{sn_id}_signals.npy"), data)
    np.save(os.path.join(OUTPUT_DIR, f"{sn_id}_labels.npy"), labels)

    print(f"✅ Saved {sn_id}: signals {data.shape}, labels {labels.shape}, unique labels {np.unique(labels)}")


def main():
    # Collect all subjects (SNxxx.edf)
    subjects = sorted([
        f.replace(".edf", "")
        for f in os.listdir(BASE_DIR)
        if f.startswith("SN") and f.endswith(".edf") and "_sleepscoring" not in f and not f.startswith("._")
    ])

    print(f"Found {len(subjects)} subjects.")
    for sn_id in subjects:
        preprocess_subject(sn_id)


if __name__ == "__main__":
    main()
