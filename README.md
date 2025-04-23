# Things-EEG Data Processing

This project processes the Things-EEG dataset using MNE and BrainDecode for further analysis.

## Dataset

The Things-EEG dataset contains EEG recordings from 50 subjects viewing 22,248 images from 1,854 object concepts in a rapid serial visual presentation (RSVP) paradigm.

Reference:
- Grootswagers, T., Zhou, I., Robinson, A.K. et al. Human EEG recordings for 1,854 concepts presented in rapid serial visual presentation streams. Sci Data 9, 3 (2022). https://doi.org/10.1038/s41597-021-01102-7

## Setup

1. Install the required dependencies:
```
pip install -r requirements.txt
```

2. Ensure the Things-EEG dataset is available in the `data/things-eeg` directory.

## Usage

1. Configure the script by editing the variables at the top of `process_things_eeg.py`:
   - `SUBJECT_IDS`: List of subject IDs to process (e.g., ['01', '02'])
   - `OUTPUT_DIR`: Directory to save processed data

2. Run the processing script:
```
python process_things_eeg.py
```

3. The script will:
   - Load the raw EEG data using MNE
   - Apply preprocessing (filtering, ICA for artifact removal)
   - Convert to BrainDecode-compatible format
   - Apply BrainDecode-specific preprocessing
   - Save the processed data in the specified output directory

## Output

The script generates the following files for each subject:
- `sub-XX_X.npy`: EEG signal data (n_epochs × n_channels × n_times)
- `sub-XX_y.npy`: Target/label data (concept IDs)
- `sub-XX_metadata.csv`: Metadata for each epoch (subject ID, concept information, etc.)

If multiple subjects are processed, a concatenated dataset is also created.

## Preprocessing Steps

1. MNE Preprocessing:
   - Re-reference to average reference
   - Bandpass filter (0.5-40 Hz)
   - ICA for artifact removal (especially eye movements)

2. BrainDecode Preprocessing:
   - Exponential moving standardization
   - Scaling to zero mean and unit variance 