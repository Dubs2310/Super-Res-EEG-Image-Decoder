import mne

def preprocess(raw: mne.io.RawArray) -> tuple:
    n_components = 6
    method = 'fastica'
    random_state = 42
    l_freq = 1.0
    h_freq = 50.0
    filter_length = 'auto'
    fir_design = 'firwin'
    ref_channels = 'average'
    ch_names = raw.ch_names
    eog_ch_names = ['AF3', 'F7', 'F8', 'AF4', 'F3', 'F4']
    eog_indices = [ch_names.index(ch) for ch in eog_ch_names]
    ch_types = ['eog' if i in eog_indices else 'eeg' for i in range(len(ch_names))]
    raw.set_channel_types(dict(zip(ch_names, ch_types)))
    raw = filter_data(raw, l_freq, h_freq, filter_length, fir_design)
    raw.set_eeg_reference(ref_channels=ref_channels)
    raw = create_and_apply_ica(raw, n_components, method, random_state)
    return raw

def filter_data(raw, l_freq, h_freq, filter_length, fir_design):
    raw.filter(l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth='auto', h_trans_bandwidth='auto', filter_length=filter_length, fir_design=fir_design)
    return raw

def create_and_apply_ica(raw, n_components, method, random_state):
    ica = mne.preprocessing.ICA(n_components=n_components, method=method, random_state=random_state, max_iter=16000)
    ica.fit(raw)
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, reject_by_annotation=False)
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
    ica.exclude += eog_inds
    raw_clean = ica.apply(raw)
    return raw_clean