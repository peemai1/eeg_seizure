import mne
import numpy as np

def load_edf_data(edf_path, preload=True):
    """
    Loads EEG data from an EDF file using MNE.
    
    Args:
        edf_path (str): Path to the .edf file.
        preload (bool): Whether to load the data into memory immediately.

    Returns:
        raw (mne.io.Raw): MNE Raw object containing EEG data.
        data (np.ndarray): Numpy array of shape (channels, samples).
        sfreq (float): Sampling frequency.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=False)
    return raw