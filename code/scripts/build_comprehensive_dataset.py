import os
import sys
import mne
import h5py
import numpy as np
from math import ceil

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', '..', 'data', 'all-joined-1')
preprocessed_data_dir = os.path.join(data_dir, 'eeg', 'preprocessed')
preprocessed_files = os.listdir(preprocessed_data_dir)
dataset_file = "comprehensive_dataset.h5"

if not preprocessed_files:
    raise FileNotFoundError('The preprocessed data directory provided has no preprocessed files.')

if os.path.exists(os.path.join(data_dir, dataset_file)):
    raise FileExistsError('Comprehensive dataset hdf5 file already exists.')
    
with h5py.File(os.path.join(data_dir, dataset_file), 'w') as f:
    pass

first_raw = mne.io.read_raw_fif(os.path.join(preprocessed_data_dir, preprocessed_files[0]), preload=True)
first_raw.drop_channels(['Status'])
sfreq = first_raw.info['sfreq']
ch_names = first_raw.info['ch_names']

epoch_config = [
    { 'mode': 'fixed_length_event', 'durations': [60, 30, 10]  },
    { 'mode': 'evoked_event', 'duration_before': 0.05, 'duration_after': 0.6 },
]

with h5py.File(os.path.join(data_dir, dataset_file), 'r+') as f:

    f.create_dataset('sfreq', data=sfreq)
    dt = h5py.special_dtype(vlen=str)
    f.create_dataset('ch_names', data=np.array(ch_names, dtype=dt))

    for config in epoch_config:
            
        if config['mode'] == 'fixed_length_event':
            for dur in config['durations']:
                f.create_dataset(
                    f'all_{dur}s_epochs',
                    shape=(0, len(ch_names), int(sfreq * dur)),
                    maxshape=(None, len(ch_names), int(sfreq * dur)),
                    dtype=np.float32
                )
                f.create_dataset(
                    f'all_{dur}s_epochs_metadata', 
                    shape=(0, 3),  # subject, session, sample_number
                    maxshape=(None, 3),
                    dtype=np.int32
                )
                    
        elif config['mode'] == 'evoked_event':
                timesteps = int(ceil(sfreq * (config['duration_before'] + config['duration_after'])))
                num_channels = len(ch_names)
                f.create_dataset(
                    'all_evoked_event_epochs',
                    shape=(0, num_channels, timesteps),
                    maxshape=(None, num_channels, timesteps),
                    dtype=np.float32
                )
                f.create_dataset(
                    'all_evoked_event_epochs_metadata', 
                    shape=(0, 4),  # subject, session, sample_number, evoked_event_id (coco train image)
                    maxshape=(None, 4),
                    dtype=np.int32
                )

for file in preprocessed_files:
    subject = file[5:6]
    session = file[14:15]

    raw = mne.io.read_raw_fif(os.path.join(preprocessed_data_dir, file), preload=True)
    
    with h5py.File(os.path.join(data_dir, dataset_file), 'r+') as f:

        for config in epoch_config:
            
            if config['mode'] == 'fixed_length_event':

                for dur in config['durations']:
                    epochs = mne.make_fixed_length_epochs(raw, duration=dur, preload=True)
                    epochs.drop_channels(['Status'])

                    data = epochs.get_data() # (batch size, channels, timesteps)
                    sample_numbers = epochs.events[:, 0] # (batch size, channels, timesteps)
                    
                    metadata = np.zeros((data.shape[0], 3), dtype=np.int32)
                    metadata[:, 0] = subject
                    metadata[:, 1] = session
                    metadata[:, 2] = sample_numbers

                    current_size = f[f'all_{dur}s_epochs'].shape[0]
                    new_size = current_size + data.shape[0]
                    f[f'all_{dur}s_epochs'].resize(new_size, axis=0)
                    f[f'all_{dur}s_epochs_metadata'].resize(new_size, axis=0)

                    f[f'all_{dur}s_epochs'][current_size:new_size] = data
                    f[f'all_{dur}s_epochs_metadata'][current_size:new_size] = metadata
                    
            if config['mode'] == 'evoked_event':

                    timesteps = int(ceil(sfreq * (config['duration_before'] + config['duration_after'])))
                    
                    evoked_events = mne.find_events(raw)
                    epochs = mne.Epochs(raw, evoked_events, tmin=-config['duration_before'], tmax=config['duration_after']+0.01, preload=True)
                    epochs.drop_channels(['Status'])

                    data = epochs.get_data()[:, :, :timesteps] # (batch size, channels, timesteps (forced))
                    sample_numbers = evoked_events[:, 0]
                    evoked_event_ids = evoked_events[:, -1]
                    
                    n_epochs = data.shape[0]
                    metadata = np.zeros((n_epochs, 4), dtype=np.int32)
                    metadata[:, 0] = subject
                    metadata[:, 1] = session
                    metadata[:, 2] = sample_numbers[:n_epochs]
                    metadata[:, 3] = evoked_event_ids[:n_epochs]

                    current_size = f['all_evoked_event_epochs'].shape[0]
                    new_size = current_size + data.shape[0]
                    f['all_evoked_event_epochs'].resize(new_size, axis=0)
                    f['all_evoked_event_epochs_metadata'].resize(new_size, axis=0)

                    f['all_evoked_event_epochs'][current_size:new_size] = data
                    f['all_evoked_event_epochs_metadata'][current_size:new_size] = metadata