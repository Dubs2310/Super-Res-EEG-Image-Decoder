import os
import mne
import argparse
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_dir', type=str, help="Where to find the raw files")
parser.add_argument('--preprocessed_data_dir', type=str, help="Where to save the preprocessed files")
parser.add_argument('--overwrite', type=bool, help='Overwrite files in the preprocessed directory')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(script_dir, '..', '..', 'data', 'all-joined-1', 'eeg', 'raw') if not args.raw_data_dir else args.raw_data_dir
preprocessed_data_dir = os.path.join(script_dir, '..', '..', 'data', 'all-joined-1', 'eeg', 'preprocessed') if not args.preprocessed_data_dir else args.preprocessed_data_dir
files = os.listdir(raw_data_dir)

if not files:
    raise FileNotFoundError('The raw data directory provided has no raw files')

# def get_events_between_timesteps_from_raw_eeg_data(subject_number, session_number, start, end):
#     raw = mne.io.read_raw_fif(os.path.join(data_dir, f'subj0{subject_number}_session{session_number}_eeg.fif'), preload=True)
#     raw = raw.crop(tmin=start / raw.info['sfreq'], tmax=end / raw.info['sfreq'])
#     return mne.find_events(raw)

# for 60 seconds
# break into fixed length of 60 seconds time windows
# _60s_epochs = mne.make_fixed_length_epochs(cropped, duration=60, preload=True)
# _30s_epochs = mne.make_fixed_length_epochs(cropped, duration=30, preload=True)
# _10s_epochs = mne.make_fixed_length_epochs(cropped, duration=10, preload=True)
# _evoked_event_epochs = mne.Epochs(cropped, all_events, preload=True, tmin=-0.05, tmax=0.6)
# _60s_epochs.get_data().shape, _30s_epochs.get_data().shape, _10s_epochs.get_data().shape, _evoked_event_epochs.get_data().shape

for file in files:

    # don't read any file other than the raw .fif file
    if not file.endswith('_eeg.fif'):
        continue
    
    # check if the file has been preprocessed before, and if it has, then ignore
    preprocessed_file_path = os.path.join(preprocessed_data_dir, file)
    if os.path.exists(preprocessed_file_path) and not args.overwrite:
        continue

    # read the raw file
    # apply bandpass and notch filters
    # apply ica
    # crop to first and last events
    # and save to preprocessed directory
    try:
        raw = mne.io.read_raw_fif(os.path.join(raw_data_dir, file), preload=True)

        filtered = raw.copy()
        montage = mne.channels.make_standard_montage('standard_1020')
        filtered.set_montage(montage)
        filtered.set_eeg_reference('average')
        filtered.filter(l_freq=0.5, h_freq=95)
        filtered.notch_filter(freqs=60)
        
        ica_cleaned = filtered.copy()
        ica = mne.preprocessing.ICA(n_components=.95, random_state=97)
        ica = ica.fit(ica_cleaned)
        ica.exclude = [1]
        ica_cleaned = ica.apply(ica_cleaned)

        # 1. First extract events and annotations
        events = mne.find_events(ica_cleaned, stim_channel='Status')
        original_annotations = ica_cleaned.annotations.copy()
        picks = mne.pick_types(ica_cleaned.info, eeg=True, stim=False)

        # 2. Z-score only EEG channels (excluding stim channels)
        ica_cleaned = ica_cleaned.get_data(picks=picks)
        ica_cleaned = (ica_cleaned - ica_cleaned.mean(axis=1, keepdims=True)) / ica_cleaned.std(axis=1, keepdims=True)

        # 3. Create normalized raw with original events
        normalized = raw.copy()
        normalized._data[picks] = ica_cleaned
        normalized.set_annotations(original_annotations)

        # fig1, fig2, fig3 = raw.plot(show=True), filtered.plot(show=True), ica_cleaned.plot(show=True)
        # normalized = zscore_normalize_raw(ica_cleaned.copy())
        all_events = mne.find_events(normalized)
        first_event_time = all_events[0, 0] / normalized.info['sfreq'] - 0.05  # 50ms before first event
        last_event_time = all_events[-1, 0] / normalized.info['sfreq'] + 0.6   # 600ms after last event
        cropped = normalized.copy().crop(tmin=first_event_time, tmax=last_event_time)

        cropped.save(preprocessed_file_path, overwrite=True)

    except Exception as e:
        print('Error with file', file, ':', e)