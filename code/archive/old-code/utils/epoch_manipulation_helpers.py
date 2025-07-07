import mne
import math
import numpy as np
from utils.coco_data_handler import COCODataHandler

def epoch_around_events(raw: mne.io.Raw, tmin, tmax, channel_names=None, resample=None):
    picks = mne.pick_channels(raw.info['ch_names'], channel_names) if channel_names is not None else mne.pick_types(raw.info, eeg=True, stim=False)
    sfreq = raw.info['sfreq']
    events = mne.find_events(raw, stim_channel="Status")
    epochs = mne.Epochs(raw, events, tmin=-tmin, tmax=tmax+0.01, picks=picks, preload=True)

    if resample is not None and resample != sfreq:
        epochs = epochs.resample(resample)
        sfreq = resample

    timesteps = int(math.ceil(sfreq * (-epochs.tmin + epochs.tmax)))
    new_tmax = epochs.tmin + (timesteps - 1) / sfreq
    epochs.crop(tmin=epochs.tmin, tmax=new_tmax)

    one_hot_encodings = np.array([COCODataHandler.get_instance()(id-1) for id in epochs.events[:, -1]])
    return epochs, one_hot_encodings


def epoch_fixed_lengths(raw: mne.io.Raw, duration, channel_names=None, resample=None):
    picks = mne.pick_channels(raw.info['ch_names'], channel_names) if channel_names is not None else mne.pick_types(raw.info, eeg=True, stim=False)
    sfreq = raw.info['sfreq']
    epochs = mne.make_fixed_length_epochs(raw, duration, preload=True).pick(picks)

    if resample is not None and resample != sfreq:
        epochs = epochs.resample(resample)
    
    return epochs, None


def reconstruct_raw_from_fixed_length_epochs(fixed_length_epochs, raw: mne.io.Raw, channel_names=None):
    total_timesteps_in_seconds = fixed_length_epochs.get_data().shape[0] * fixed_length_epochs.get_data().shape[-1] / fixed_length_epochs.info['sfreq']
    copy = raw.load_data().copy().crop(tmax=total_timesteps_in_seconds)

    if channel_names is not None:
        picks = mne.pick_channels(copy.info['ch_names'], channel_names)
        copy.pick(picks)

    ica = mne.preprocessing.ICA(random_state=97)
    ica.fit(fixed_length_epochs)
    return ica.apply(copy, include=[])