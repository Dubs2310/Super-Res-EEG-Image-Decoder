import mne
import math

def epoch_around_events(raw: mne.io.Raw, tmin, tmax, stim_channel="Status", picks=None, baseline=None, resample=None, crop_to_exact_samples=False, crop_buffer=0.1, event_id=None, preload=True, reject=None, flat=None):
    if picks is None:
        picks = mne.pick_types(raw.info, eeg=True, stim=False)

    epoch_tmax = tmax + crop_buffer if crop_to_exact_samples else tmax
    events = mne.find_events(raw, stim_channel=stim_channel)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-tmin, tmax=epoch_tmax, picks=picks, preload=preload, baseline=baseline, reject=reject, flat=flat)
    dropped_epoch_indices = [ i for i, drop_log in enumerate(epochs.drop_log) if len(drop_log) > 0 ]
    
    if resample is not None and resample != raw.info['sfreq']:
        epochs = epochs.resample(resample)
    
    if crop_to_exact_samples:
        sfreq = epochs.info['sfreq']
        n_samples = int(math.ceil(sfreq * (tmin + tmax)))
        exact_tmax = epochs.tmin + (n_samples - 1) / sfreq
        epochs.crop(tmin=epochs.tmin, tmax=exact_tmax)
    
    return epochs, dropped_epoch_indices
 
def crop_epoch(epoch, original_window_before_ms, new_window_before_ms, new_window_after_ms, sfreq):
   original_before_samples = int(original_window_before_ms * sfreq / 1000)
   new_before_samples = int(new_window_before_ms * sfreq / 1000)
   new_after_samples = int(new_window_after_ms * sfreq / 1000)
   start_idx = original_before_samples - new_before_samples
   end_idx = start_idx + new_before_samples + new_after_samples
   cropped_epoch = epoch[:, start_idx:end_idx]
   return cropped_epoch