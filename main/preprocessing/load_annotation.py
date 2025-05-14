import pandas as pd

def load_annotation(tsv_path, sfreq=128):
    """
    Parses the TSV annotation file and returns a list of seizure intervals in sample units.

    Parameters:
        tsv_path (str): Path to the TSV file.
        sfreq (int): Sampling frequency (e.g., 128 Hz).

    Returns:
        List of tuples: [(start_sample, end_sample), ...]
    """
    df = pd.read_csv(tsv_path, sep='\t')

    seizure_intervals = []
    for _, row in df.iterrows():
        if 'sz' in row['eventType'].lower():
            onset_sec = float(row['onset'])
            duration_sec = float(row['duration'])
            start_sample = int(onset_sec * sfreq)
            end_sample = int((onset_sec + duration_sec) * sfreq)
            seizure_intervals.append((start_sample, end_sample))

    return seizure_intervals