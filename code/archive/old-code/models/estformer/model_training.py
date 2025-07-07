# %%
import os
import sys
import torch
import wandb
import GPUtil
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ESTFormer import ESTFormer

sys.path.append('../../')
from utils.epoch_data_reader import EpochDataReader

# %%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Force CUDA to use the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Enable memory optimization settings for PyTorch

# %%
# Check if CUDA is available
try:
    gpus = GPUtil.getGPUs()
    if gpus:
        print(f"GPUtil detected {len(gpus)} GPUs:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name} (Memory: {gpu.memoryTotal}MB)")
        
        # Set default GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(len(gpus))])
        print(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("GPUtil found no available GPUs")
except Exception as e:
    print(f"Error checking GPUs with GPUtil: {e}")

# %%
# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Print available GPU memory
if torch.cuda.is_available():
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# %%
all_channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

# Model parameters
hr_channel_names = all_channels # High-resolution setup (all channels)
lr_channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'] # Low-resolution setup (fewer channels)
builtin_montage = 'standard_1020'
alpha_t = 0.60
alpha_s = 0.75
r_mlp = 4 # amplification factor for MLP layers
dropout_rate = 0.5
L_s = 1  # Number of spatial layers
L_t = 1  # Number of temporal layers

# Training parameters
epochs = 1

# Optimizer parameters
lr = 5e-5
weight_decay = 0.5
beta_1 = 0.9
beta_2 = 0.95

# Dataset parameters
# split = "70/25/5"
# epoch_type = "around_evoked"
# before = 0.05
# after = 0.6
# random_state = 97

# Data Loader parameter
batch_size = 30

# %%
# Create datasets
lo_res_dataset = EpochDataReader(
    channel_names=lr_channel_names
)

hi_res_dataset = EpochDataReader(
    channel_names=hr_channel_names
)

# %%
lo_res_loader = DataLoader(
    lo_res_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

hi_res_loader = DataLoader(
    hi_res_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

len(lo_res_loader), len(hi_res_loader)

# %%
# Get sample data to determine time_steps
sample_item = lo_res_dataset[0][0] if lo_res_dataset.epoch_type == 'around_evoked' else lo_res_dataset[0]
time_steps = sample_item.shape[1]
sfreq = lo_res_dataset.resample_freq

config = {
    "total_epochs_trained_on": epochs,
    "scale_factor": len(hr_channel_names) / len(lr_channel_names),
    "time_steps_in_seconds": time_steps / sfreq,
    "is_parieto_occipital_exclusive": all(ch.startswith('P') or ch.startswith('O') or ch.startswith('PO') or ch.startswith('CP') for ch in lr_channel_names) and all(ch.startswith('P') or ch.startswith('O') or ch.startswith('PO') or ch.startswith('CP') for ch in hr_channel_names),
    "model_params": {
        "model": "ESTformer",
        "num_lr_channels": len(lr_channel_names),
        "num_hr_channels": len(hr_channel_names),
        "builtin_montage": builtin_montage,
        "alpha_s": alpha_s,
        "alpha_t": alpha_t,
        "r_mlp": r_mlp,
        "dropout_rate": dropout_rate,
        "L_s": L_s,
        "L_t": L_t,
    },
    "dataset_params": {
        "subject_session_id": lo_res_dataset.subject_session_id,
        "epoch_type": lo_res_dataset.epoch_type,
        "split": lo_res_dataset.split,
        "duration": str((lo_res_dataset.before + lo_res_dataset.after) * 1000) + 'ms' if lo_res_dataset.epoch_type == 'around_evoked' else lo_res_dataset.fixed_length_duration,
        "batch_size": batch_size,
        "random_state": lo_res_dataset.random_state
    },
    "optimizer_params": {
        "optimizer": "Adam",
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "betas": (beta_1, beta_2)
    }
}

# %%
model = ESTFormer(
    device=device, 
    lr_channel_names=lr_channel_names,
    hr_channel_names=hr_channel_names,
    builtin_montage=builtin_montage,
    time_steps=time_steps,
    alpha_t=alpha_t,
    alpha_s=alpha_s,
    r_mlp=r_mlp,
    dropout_rate=dropout_rate,
    L_s=L_s,
    L_t=L_t
)

summary(model)

# %%
# Create optimizer with both model and sigma parameters
optimizer = optim.Adam(
    params=[{'params': model.parameters()}], 
    lr=lr,
    weight_decay=weight_decay,
    betas=(beta_1, beta_2)
)

with wandb.init(project="eeg-estformer", config=config) as run:
    history = model.fit(
        epochs=epochs,
        lo_res_loader=lo_res_loader,
        hi_res_loader=hi_res_loader,
        optimizer=optimizer,
        checkpoint_dir='checkpoints',
        identifier='test'
    )


# %%
# average_test_results = model.predict(test_loader)
# print("Average Results on Test Set: ", average_test_results)

# %%
# def monitor_sigma_values_and_loss(history):
#     """
#     Monitor the values of sigma1 and sigma2 during training.
    
#     Args:
#         history: Training history dictionary
#     """
#     # Get the values of sigma1 and sigma2
#     sigma1_values = history['sigma1']
#     sigma2_values = history['sigma2']
    
#     print(f"Final sigma1 value: {sigma1_values[-1]}")
#     print(f"Final sigma2 value: {sigma2_values[-1]}")
    
#     # Plot the loss history
#     plt.figure(figsize=(12, 8))
    
#     # Plot loss
#     plt.subplot(2, 2, 1)
#     plt.plot(history['train_loss'], label='Training Loss')
#     plt.plot(history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
    
#     # Plot MAE
#     plt.subplot(2, 2, 2)
#     plt.plot(history['train_mae'], label='Training MAE')
#     plt.plot(history['val_mae'], label='Validation MAE')
#     plt.title('Model MAE')
#     plt.xlabel('Epoch')
#     plt.ylabel('MAE')
#     plt.legend()

#     # Plot NMSE
#     plt.subplot(2, 2, 3)
#     plt.plot(history['train_nmse'], label='Training NMSE')
#     plt.plot(history['val_nmse'], label='Validation NMSE')
#     plt.title('Model NMSE')
#     plt.xlabel('Epoch')
#     plt.ylabel('NMSE')
#     plt.legend()

#     # Plot SNR
#     plt.subplot(2, 2, 4)
#     plt.plot(history['train_snr'], label='Training SNR')
#     plt.plot(history['val_snr'], label='Validation SNR')
#     plt.title('Model SNR')
#     plt.xlabel('Epoch')
#     plt.ylabel('SNR')
#     plt.legend()
    
#     # Plot PCC
#     plt.subplot(2, 2, 5)
#     plt.plot(history['train_pcc'], label='Training PCC')
#     plt.plot(history['val_pcc'], label='Validation PCC')
#     plt.title('Model PCC')
#     plt.xlabel('Epoch')
#     plt.ylabel('PCC')
#     plt.legend()
    
#     # Plot sigma values
#     plt.subplot(2, 2, 3)
#     plt.plot(sigma1_values, label='Sigma1')
#     plt.title('Sigma1 Value')
#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
    
#     plt.subplot(2, 2, 4)
#     plt.plot(sigma2_values, label='Sigma2')
#     plt.title('Sigma2 Value')
#     plt.xlabel('Epoch')
#     plt.ylabel('Value')
    
#     plt.tight_layout()
    
#     # Save figure to wandb
#     if wandb.run is not None:
#         wandb.log({"training_history": wandb.Image(plt)})
    
#     plt.show()

# monitor_sigma_values_and_loss(history)

# %%
# def visualize_results(model, val_dataset, device, subject_idx=0, channel_idx=0):
#     """
#     Visualize the results of the model on a validation sample.
    
#     Args:
#         model: Trained ESTformer model
#         val_dataset: Validation dataset
#         device: Device to run inference on
#         subject_idx: Index of the subject to visualize
#         channel_idx: Index of the channel to visualize
#     """
#     # Set model to eval mode
#     model.eval()
    
#     # Get a validation sample
#     sample = val_dataset[subject_idx]
    
#     # Convert to tensors and add batch dimension
#     lo_res = torch.tensor(sample['lo_res'], dtype=torch.float32).unsqueeze(0).to(device)
#     hi_res = torch.tensor(sample['hi_res'], dtype=torch.float32)
    
#     # Get predictions
#     with torch.no_grad():
#         pred = model(lo_res).cpu().numpy()[0]
    
#     # Convert back to numpy for visualization
#     lo_res = lo_res.cpu().numpy()[0]
#     hi_res = hi_res.numpy()
    
#     # Plot the results
#     plt.figure(figsize=(12, 8))
    
#     # Plot low-res input
#     plt.subplot(3, 1, 1)
#     plt.plot(lo_res[channel_idx])
#     plt.title(f'Low-Res (Downsampled) Input (Channel {channel_idx})')
    
#     # Plot high-res ground truth
#     plt.subplot(3, 1, 2)
#     plt.plot(hi_res[channel_idx])
#     plt.title(f'High-Res (Ground Truth) (Channel {channel_idx})')
    
#     # Plot prediction
#     plt.subplot(3, 1, 3)
#     plt.plot(pred[channel_idx])
#     plt.title(f'Super-Res (Prediction) (Channel {channel_idx})')
    
#     plt.tight_layout()
    
#     # Save figure to wandb
#     if wandb.run is not None:
#         wandb.log({"prediction_visualization": wandb.Image(plt)})
    
#     plt.show()

# visualize_results(model, val_loader.dataset, device)

