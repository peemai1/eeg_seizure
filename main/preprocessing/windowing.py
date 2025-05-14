import numpy as np

def create_windows(data, seizure_intervals, sfreq=128, window_sec=5, overlap=2.5):
    """
    Splits multichannel EEG data into overlapping windows with per-sample labels.

    Parameters:
    - data: np.array shape (n_channels, n_samples)
    - seizure_intervals: list of tuples [(start_sample, end_sample), ...]
    - sfreq: sampling frequency (e.g., 128 Hz)
    - window_sec: window size in seconds
    - stride_sec: stride size in seconds (how far window moves)
    
    Returns:
    - windows: shape [num_windows, n_channels, window_samples]
    - labels: shape [num_windows, window_samples] — binary label per sample
    """

    window_size = int(window_sec * sfreq)
    stride = int(overlap * sfreq)
    total_samples = data.shape[1]

    windows = []
    labels = []

    for start in range(0, total_samples - window_size + 1, stride):
        end = start + window_size
        window = data[:, start:end]
        label = np.zeros(window_size)

        # Check overlap with any seizure interval
        for sz_start, sz_end in seizure_intervals:
            overlap_start = max(start, sz_start)
            overlap_end = min(end, sz_end)
            if overlap_start < overlap_end:
                label[overlap_start - start : overlap_end - start] = 1

        windows.append(window)
        labels.append(label)

    return np.array(windows), np.array(labels)
