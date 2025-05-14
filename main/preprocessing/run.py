import os
import glob
import mne
import numpy as np
from filter_utils import (rename_channels_standard, apply_montage,apply_filters, run_ica_artifact_removal)
from windowing import create_windows
from load_annotation import load_annotation

def preprocess_subject(edf_path, ann_path, output_dir, subject_id):
    """
    Full ETL pipeline for a single subject: loads EDF + annotations,
    filters, resamples, segments into windows, and saves labeled windows.

    Parameters:
        edf_path (str): Path to the .edf EEG file.
        ann_path (str): Path to the .tsv annotation file.
        output_dir (str): Where to save the .npz output.
        subject_id (str): Identifier (e.g., 'sub-00').
        sfreq (int): Target sampling frequency.
        window_size_sec (int): Length of each window in seconds.
        overlap (float): Proportion of overlap between windows (0-1).
    """

    print(f"▶️  Processing {subject_id} | {edf_path}")

    # --- Load and clean EEG ---
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    prep = rename_channels_standard(raw)
    prep = apply_montage(prep)
    prep = apply_filters(prep) 
    prep = run_ica_artifact_removal(prep)

    # --- Extract numpy data ---
    data = prep.get_data()  # shape: (19, n_samples)

    # --- Load seizure annotations ---
    seizure_intervals = load_annotation(ann_path)

    # --- Create windows + labels ---
    windows, labels = create_windows(data, seizure_intervals)

    # --- Save as .npz file ---
    # save_path = os.path.join(output_dir, f"{subject_id}.npz")
    # np.savez_compressed(save_path, X=windows, y=labels, subject_id=subject_id)

    # print(f" Done: {subject_id} path {save_path} : -> {save_path} ({windows.shape[0]} windows)")
    
    if output_dir:
        save_path = os.path.join(output_dir, f"{subject_id}_runX.npz")  # optional
        np.savez_compressed(save_path, X=windows, y=labels, subject_id=subject_id)
        print(f"✅ Saved run: {save_path}")
    else:
        return windows, labels


# === Run pipeline for all subjects === #
# Paths
raw_root = '/Users/peemaisuakaew/Desktop/eeg_seizure/data/raw/BIDS_Siena/'
annotation_root = '/Users/peemaisuakaew/Desktop/eeg_seizure/data/annotations/'
output_root = '/Users/peemaisuakaew/Desktop/eeg_seizure/data/processed/cleaned/' # 'cleaned' or 'uncleaned'

# Create output dir
# os.makedirs(output_root, exist_ok=True)

# Find all .edf files
edf_paths = sorted(glob.glob(os.path.join(raw_root, 'sub-*', '*_eeg.edf')))

# Group by subject
subject_data = {}
for edf_path in edf_paths:
    base = os.path.basename(edf_path)
    subject_id = base.split('_')[0]  

    # Match annotation
    ann_path = os.path.join(annotation_root, subject_id, base.replace('_eeg.edf', '_events.tsv'))
    if not os.path.exists(ann_path):
        print(f"⚠️ Missing annotation for {base}, skipping.")
        continue

    # Run ETL for this run and return data
    try:
        X, y = preprocess_subject(
            edf_path=edf_path,
            ann_path=ann_path,
            output_dir=None,  # <-- disable saving per-run
            subject_id=subject_id,
        )
    except Exception as e:
        print(f"❌ Failed {edf_path}: {e}")
        continue

    # Add to subject group
    if subject_id not in subject_data:
        subject_data[subject_id] = {"X": [], "y": []}
    subject_data[subject_id]["X"].append(X)
    subject_data[subject_id]["y"].append(y)

# Save one .npz per subject with 'run_ids'
for subject_id, data in subject_data.items():
    X_all = np.concatenate(data["X"], axis=0)
    y_all = np.concatenate(data["y"], axis=0)

    run_lengths = [x.shape[0] for x in data["X"]]
    run_ids = np.concatenate([ np.full(length, idx) for idx, length in enumerate(run_lengths)])

    save_path = os.path.join(output_root, f"{subject_id}.npz")
    np.savez_compressed(
        save_path,
        X=X_all,
        y=y_all,
        subject_id=subject_id,
        run_ids=run_ids
    )
    print(f"✅ Saved {subject_id} : {X_all.shape[0]} windows, {len(run_lengths)} runs")