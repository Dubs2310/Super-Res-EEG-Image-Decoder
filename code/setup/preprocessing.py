import os
import mne
import argparse
mne.set_log_level('WARNING')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, help="Where to find the raw files")
    parser.add_argument('--preprocessed_data_dir', type=str, help="Where to save the preprocessed files")
    parser.add_argument('--overwrite', action='store_true', help='Overwrite files in the preprocessed directory')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(script_dir, '..', '..', 'data', 'all-joined-1', 'eeg', 'raw') if not args.raw_data_dir else args.raw_data_dir
    preprocessed_data_dir = os.path.join(script_dir, '..', '..', 'data', 'all-joined-1', 'eeg', 'preprocessed') if not args.preprocessed_data_dir else args.preprocessed_data_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    
    # Check if raw data directory exists
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f'The raw data directory "{raw_data_dir}" does not exist')
    
    files = os.listdir(raw_data_dir)
    
    if not files:
        raise FileNotFoundError('The raw data directory provided has no raw files')

    # Filter for EEG files only
    eeg_files = [f for f in files if f.endswith('_eeg.fif')]
    
    if not eeg_files:
        raise FileNotFoundError('No EEG files found in the raw data directory')
    
    print(f"Found {len(eeg_files)} EEG files to process")
    
    for file in eeg_files:
        print(f"\nProcessing {file}...")
        
        # Check if the file has been preprocessed before
        preprocessed_file_path = os.path.join(preprocessed_data_dir, file)
        
        if os.path.exists(preprocessed_file_path) and not args.overwrite:
            print(f"  Skipping {file} (already exists and overwrite=False)")
            continue
        
        if os.path.exists(preprocessed_file_path) and args.overwrite:
            print(f"  Overwriting existing file for {file}")
        
        # Preprocess the file
        try:
            raw_file_path = os.path.join(raw_data_dir, file)
            raw = mne.io.read_raw_fif(raw_file_path, preload=True)
            
            print(f"  Applying filters and preprocessing...")
            
            # Apply preprocessing steps
            filtered = raw.copy()
            montage = mne.channels.make_standard_montage('standard_1020')
            filtered.set_montage(montage)
            filtered.set_eeg_reference('average')
            filtered.filter(l_freq=0.5, h_freq=95)
            filtered.notch_filter(freqs=60)
            
            # Apply ICA
            ica_cleaned = filtered.copy()
            ica = mne.preprocessing.ICA(n_components=.95, random_state=97)
            ica = ica.fit(ica_cleaned)
            ica.exclude = [1]
            ica_cleaned = ica.apply(ica_cleaned)

            # Z-score normalization
            events = mne.find_events(ica_cleaned, stim_channel='Status')
            picks = mne.pick_types(ica_cleaned.info, eeg=True, stim=False)

            # Z-score only EEG channels (excluding stim channels)
            norm_data = ica_cleaned.get_data(picks=picks)
            norm_data = (norm_data - norm_data.mean(axis=1, keepdims=True)) / norm_data.std(axis=1, keepdims=True)

            # Create normalized raw with original events
            normalized = ica_cleaned.copy()
            normalized._data[picks] = norm_data

            # Save preprocessed file
            normalized.save(preprocessed_file_path, overwrite=True)
            print(f"  Successfully saved preprocessed file: {preprocessed_file_path}")

        except Exception as e:
            print(f"  Error processing {file}: {e}")
            continue
    
    print(f"\nPreprocessing complete!")

if __name__ == "__main__":
    main()