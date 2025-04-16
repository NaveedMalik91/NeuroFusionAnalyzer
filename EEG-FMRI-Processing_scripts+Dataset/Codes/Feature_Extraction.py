import os
import numpy as np
import pandas as pd
import mne
import nibabel as nib
from scipy.signal import welch, butter, filtfilt
from scipy.stats import entropy

print("Initializing EEG and fMRI feature extraction...")

# EEG Feature Extraction
def extract_eeg_features(eeg_file, subject, run, task):
    try:
        print(f"Processing EEG file: {eeg_file} for Subject: {subject}, Run: {run}, Task: {task}")
        raw = mne.io.read_raw_brainvision(eeg_file, preload=True)
        # Apply uniform high-pass filter (e.g., 0.1 Hz)
        raw.filter(l_freq=0.1, h_freq=None, fir_design='firwin')

        sfreq = raw.info['sfreq']
        data, _ = raw[:]

        print("Computing Power Spectral Density (PSD)...")
        freqs, psd = welch(data, sfreq, nperseg=int(sfreq * 2))  # Ensure correct segment size
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}

        band_power = {}
        for band, (low, high) in bands.items():
            band_idx = np.where((freqs >= low) & (freqs <= high))
            band_power[band] = np.mean(psd[:, band_idx], axis=1).mean()

        print("Calculating Spectral Entropy...")
        spectral_entropy = entropy(psd, axis=1).mean()

        print("Computing Mean Coherence...")
        coherence_matrix = np.corrcoef(data)
        mean_coherence = np.mean(coherence_matrix[np.triu_indices_from(coherence_matrix, k=1)])

        print("EEG feature extraction completed.")
        return [subject, run, task] + list(band_power.values()) + [spectral_entropy, mean_coherence]
    except Exception as e:
        print(f"Error processing EEG file {eeg_file}: {e}")
        return None

# fMRI Feature Extraction
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Reshaping data for correct filtering
    original_shape = data.shape
    flattened_data = data.reshape(-1, original_shape[-1])  # Reshape to apply 1D filter
    filtered = np.array([filtfilt(b, a, voxel) for voxel in flattened_data])
    return filtered.reshape(original_shape)  # Reshape back to original shape

def extract_fmri_features(fmri_file, subject, run, task):
    try:
        print(f"Processing fMRI file: {fmri_file} for Subject: {subject}, Run: {run}, Task: {task}")
        img = nib.load(fmri_file)
        fmri_data = img.get_fdata()

        print("Checking fMRI Data Shape:", fmri_data.shape)  # Debugging

        print("Calculating BOLD Mean and Variance...")
        bold_mean = np.mean(fmri_data)
        bold_variance = np.var(fmri_data)

        print("Computing ALFF Mean and Variance...")
        tr = 2.0  # Adjust TR based on dataset
        filtered_data = butter_bandpass_filter(fmri_data, 0.01, 0.1, 1.0 / tr)

        print(f"Filtered Data Shape: {filtered_data.shape}")
        print(f"Filtered Data Min: {filtered_data.min()}, Max: {filtered_data.max()}")

        alff_mean = np.mean(np.abs(filtered_data))
        alff_variance = np.var(np.abs(filtered_data))

        print("fMRI feature extraction completed.")
        return [subject, run, task, bold_mean, bold_variance, alff_mean, alff_variance]
    except Exception as e:
        print(f"Error processing fMRI file {fmri_file}: {e}")
        return None

# Define main directory
main_dir = "/content/ds003768-download"  # Adjust this path as necessary
subjects = [f"sub-{i:02d}" for i in range(11,12)]

eeg_data = []
fmri_data = []

print("Starting EEG and fMRI feature extraction for all subjects...")

for subject in subjects:
    subject_path = os.path.join(main_dir, subject)
    eeg_path = os.path.join(subject_path, "eeg")
    func_path = os.path.join(subject_path, "func")

    if not os.path.exists(eeg_path) or not os.path.exists(func_path):
        print(f"Skipping {subject} due to missing EEG or fMRI directories.")
        continue

    eeg_files = [(f, subject, int(f.split("_run-")[1][0]), f.split("_task-")[1].split("_run-")[0])
                 for f in os.listdir(eeg_path) if f.endswith(".vhdr")]
    fmri_files = [(f, subject, int(f.split("_run-")[1][0]), f.split("_task-")[1].split("_run-")[0])
                  for f in os.listdir(func_path) if f.endswith(".nii.gz")]

    for eeg_file, subject, run, task in eeg_files:
        result = extract_eeg_features(os.path.join(eeg_path, eeg_file), subject, run, task)
        if result:
            eeg_data.append(result)

    for fmri_file, subject, run, task in fmri_files:
        result = extract_fmri_features(os.path.join(func_path, fmri_file), subject, run, task)
        if result:
            fmri_data.append(result)

eeg_df = pd.DataFrame(eeg_data, columns=["Subject", "Run", "Task", "Delta Power", "Theta Power", "Alpha Power", "Beta Power", "Gamma Power", "Spectral Entropy", "Mean Coherence"])
eeg_df.to_csv("eeg_features3.csv", index=False)
print("EEG feature extraction saved to 'eeg_features3.csv'.")

fmri_df = pd.DataFrame(fmri_data, columns=["Subject", "Run", "Task", "BOLD Mean", "BOLD Variance", "ALFF Mean", "ALFF Variance"])
fmri_df.to_csv("fmri_features3.csv", index=False)
print("fMRI feature extraction saved to 'fmri_features3.csv'.")

print("EEG and fMRI feature extraction process completed successfully! 🎉")
