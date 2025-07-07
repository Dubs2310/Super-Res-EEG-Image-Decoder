import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.epoch_data_reader import EpochDataReader

class EpochDataModule(pl.LightningDataModule):
    def __init__(self, subject_session_id='cross', read_from="ground-truth", channel_names=None, resample_freq=512, epoch_type="around_evoked", before=0.05, after=0.6, fixed_length_duration=8, split_type=None, split="70/25/5", random_state=97, batch_size=32, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
        super().__init__()
        self.data_reader = EpochDataReader(subject_session_id=subject_session_id, read_from=read_from, channel_names=channel_names, resample_freq=resample_freq, epoch_type=epoch_type, before=before, after=after, fixed_length_duration=fixed_length_duration, split_type=split_type, split=split, random_state=random_state)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def train_dataloader(self):
        self.data_reader.set_split_type("train")
        loader = DataLoader(self.data_reader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last)
        return loader

    def val_dataloader(self):
        self.data_reader.set_split_type("val")
        loader = DataLoader(self.data_reader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last)
        return loader

    def test_dataloader(self):
        self.data_reader.set_split_type("test")
        loader = DataLoader(self.data_reader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last)
        return loader

class SuperResEpochDataModule(pl.LightningDataModule):
    def __init__(self, subject_session_id='cross', read_from="ground-truth", lo_res_channel_names=None, hi_res_channel_names=None, resample_freq=512, epoch_type="around_evoked", before=0.05, after=0.6, fixed_length_duration=8, split_type=None, split="70/25/5", random_state=97, batch_size=32, shuffle=False, num_workers=0, pin_memory=False, drop_last=False):
        super().__init__()
        self.lo_res_data_reader = EpochDataReader(subject_session_id=subject_session_id, read_from=read_from, channel_names=lo_res_channel_names, resample_freq=resample_freq, epoch_type=epoch_type, before=before, after=after, fixed_length_duration=fixed_length_duration, split_type=split_type, split=split, random_state=random_state)
        self.hi_res_data_reader = EpochDataReader(subject_session_id=subject_session_id, read_from=read_from, channel_names=hi_res_channel_names, resample_freq=resample_freq, epoch_type=epoch_type, before=before, after=after, fixed_length_duration=fixed_length_duration, split_type=split_type, split=split, random_state=random_state)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def train_dataloader(self):
        self.lo_res_data_reader.set_split_type("train")
        self.hi_res_data_reader.set_split_type("train")
        lo_res_loader = DataLoader(self.lo_res_data_reader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last)
        hi_res_loader = DataLoader(self.hi_res_data_reader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last)
        loader = zip(lo_res_loader, hi_res_loader)
        return loader

    def val_dataloader(self):
        self.lo_res_data_reader.set_split_type("val")
        self.hi_res_data_reader.set_split_type("val")
        lo_res_loader = DataLoader(self.lo_res_data_reader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last)
        hi_res_loader = DataLoader(self.hi_res_data_reader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last)
        loader = zip(lo_res_loader, hi_res_loader)
        return loader

    def test_dataloader(self):
        self.lo_res_data_reader.set_split_type("test")
        self.hi_res_data_reader.set_split_type("test")
        lo_res_loader = DataLoader(self.lo_res_data_reader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last)
        hi_res_loader = DataLoader(self.hi_res_data_reader, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle, pin_memory=self.pin_memory, drop_last=self.drop_last)
        loader = zip(lo_res_loader, hi_res_loader)
        return loader