import re
import numpy as np
import mne

def rename_channels_standard(raw):
    """
    Rename EDF channels to standard 10-20 names using regex mapping.
    """
    rename = {}
    for ch in raw.ch_names:
        match = re.search(r'([A-Za-z][PpZz]?[0-9]?)', ch)
        if match:
            rename[ch] = match.group().capitalize()
    raw.rename_channels(rename)
    return raw

def apply_montage(raw, montage_name="standard_1020"):
    """
    Set standard 10-20 electrode montage.
    """
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage)
    return raw

def apply_filters(raw, notch_freqs=(50, 100)):
    """
    Apply notch filter and downsample(1/2) the EEG data.
    """
    raw.notch_filter(freqs=notch_freqs)
    #raw.filter(l_freq=0.1, h_freq=45)
    raw.resample(sfreq=raw.info['sfreq'] / 2, npad="auto")
    return raw

def run_ica_artifact_removal(raw, tstep=1, random_state=42):
    """
    Run ICA to remove muscle and EOG artifacts.
    """
    # Create fixed-length events
    events = mne.make_fixed_length_events(raw, duration=tstep)
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=tstep, baseline=None, preload=True)

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=0.99, random_state=random_state)
    ica.fit(epochs, decim=3)

    # Muscle artifacts
    muscle_idx, _ = ica.find_bads_muscle(raw)
    ica.exclude = muscle_idx
    raw = ica.apply(raw)

    # EOG artifacts (assuming 'Fp1' is references to frontopolar electrode)
    _, eog_scores = ica.find_bads_eog(epochs, ch_name='Fp1', measure='zscore')
    if eog_scores is not None and len(eog_scores) > 0:
        eog_idx = np.argmax(np.abs(eog_scores))
        ica.exclude = [eog_idx]
        raw = ica.apply(raw)

    return raw
