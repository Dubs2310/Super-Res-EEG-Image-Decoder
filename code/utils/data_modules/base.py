import pytorch_lightning as pl
from utils.singletons.coco import COCO
from torch.utils.data import DataLoader
from utils.datasets.base import EEGDataset
from sklearn.model_selection import train_test_split

class EEGDataModule(pl.LightningDataModule):
    def __init__(self, dataset_class: EEGDataset, input_channels, sfreq, window_before_event_ms, window_after_event_ms, montage=None, eeg_dir=None, epochs_dir=None, subject=None, session=None, batch_size=32, num_workers=4, val_split=0.1, test='default'):
        super().__init__()
        self.dataset_class = dataset_class
        self.dataset_input_params = {
            'input_channels': input_channels,
            'sfreq': sfreq,
            'window_before_event_ms': window_before_event_ms,
            'window_after_event_ms': window_after_event_ms,
            'montage': montage,
            'eeg_dir': eeg_dir,
            'epochs_dir': epochs_dir
        }
        self.subject = subject
        self.session = session
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test = test  # New parameter: 'default' or 'All'
        self.coco = COCO()

    def _filter_dataframe(self, df):
        if self.subject and self.session:
            return df[(df['subject'] == self.subject) & (df['session'] == self.session)]
        elif self.subject and not self.session:
            return df[df['subject'] == self.subject]
        else:
            return df
    
    def get_dataset_output_params(self, df, split='train'):
        raise NotImplementedError("Subclasses must implement get_dataset_output_params")

    def get_output_sample_info(self, output_sample):
        raise NotImplementedError("Subclasses must implement get_output_sample_info")
        
    def get_sample_info(self):
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            self.setup('fit')

        if len(self.train_dataset) > 0:
            input_sample, output_sample = self.train_dataset[0]
            sample_info = {
                'input': {
                    'num_channels': input_sample.shape[0],
                    'num_timesteps': input_sample.shape[1],
                    'channel_names': self.dataset_input_params['input_channels'],
                    'sfreq': self.dataset_input_params['sfreq'],
                    'epoch_window_ms': self.dataset_input_params['window_before_event_ms'] + self.dataset_input_params['window_after_event_ms'],
                    'montage': self.dataset_input_params['montage']
                },
                'output': self.get_output_sample_info(output_sample)
            }
            return sample_info
        else:
            raise ValueError("No samples found in dataset")

    def setup(self, stage=None):
        # Check if datasets are already initialized
        if hasattr(self, 'train_dataset') and hasattr(self, 'val_dataset') and hasattr(self, 'test_dataset'):
            print("Datasets already initialized, skipping setup")
            return
            
        if self.test == 'All':
            # Use all available data for the specified subject/session as TEST SET
            print(f"Using ALL data for subject={self.subject}, session={self.session} as TEST SET")
            all_df = self.coco.get_global_eeg_image_df()
            test_df = self._filter_dataframe(all_df)
            
            if len(test_df) == 0:
                raise ValueError(f"No data found for subject={self.subject}, session={self.session}")
            
            print(f"Found {len(test_df)} total samples for test set")
            
            # For train and val, use default COCO splits but filtered
            train_df, _, _, _ = self.coco.get_train_set()
            train_df = self._filter_dataframe(train_df)
            
            if len(train_df) == 0:
                # If no train data for this subject/session, create empty datasets
                print("Warning: No training data found for this subject/session")
                train_df = test_df.iloc[:0]  # Empty dataframe with same structure
                val_df = test_df.iloc[:0]
            else:
                train_df, val_df = train_test_split(train_df, test_size=self.val_split, random_state=42)
            
            print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
        else:
            # Use default train/test split from COCO
            print("Using default COCO train/test split")
            train_df, _, _, _ = self.coco.get_train_set()
            test_df, _, _, _ = self.coco.get_test_set()
            train_df = self._filter_dataframe(train_df)
            test_df = self._filter_dataframe(test_df)
            
            train_df, val_df = train_test_split(train_df, test_size=self.val_split, random_state=42)
            
            print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        print('Creating Datasets...')
        train_output_args = self.get_dataset_output_params(train_df, 'train')
        val_output_args = self.get_dataset_output_params(val_df, 'val')
        
        # Pass 'all' split for test dataset when using all data
        test_split = 'all' if self.test == 'All' else 'test'
        test_output_args = self.get_dataset_output_params(test_df, test_split)
        
        # Create datasets
        self.train_dataset = self.dataset_class(train_df, **self.dataset_input_params, **train_output_args)
        self.val_dataset = self.dataset_class(val_df, **self.dataset_input_params, **val_output_args)
        self.test_dataset = self.dataset_class(test_df, **self.dataset_input_params, **test_output_args)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)