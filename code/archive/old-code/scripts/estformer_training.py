import os
import sys
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import GPUtil
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.estformer.ESTFormer import ESTFormer, reconstruction_loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Force CUDA to use the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # Enable memory optimization settings for PyTorch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", '..'))
from utils.hdf5_data_split_generator import HDF5DataSplitGenerator
from utils.metrics import (
    mae as compute_mean_absolute_error, 
    nmse as compute_normalized_mean_squared_error, 
    pcc as compute_pearson_correlation_coefficient,
    snr as compute_signal_to_noise_ratio
)

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

class SigmaParameters(nn.Module):
    """Class to hold trainable sigma parameters"""
    def __init__(self):
        super().__init__()
        self.sigma1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.sigma2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

def create_estformer_model(lr_channel_names, hr_channel_names, time_steps, alpha_s=0.75, alpha_t=0.60, r_mlp=128, dropout_rate=0.5, L_s=1, L_t=1, builtin_montage='standard_1020'):
    """
    Create the ESTformer model with trainable sigma parameters.
    
    Args:
        lr_channel_names: List of low-resolution channel names
        hr_channel_names: List of high-resolution channel names
        time_steps: Number of time steps in each epoch
        alpha_s: Dimension of the model
        r_mlp: Dimension of MLP layers
        dropout_rate: Dropout rate
        L_s: Number of spatial layers
        L_t: Number of temporal layers
        builtin_montage: Name of the montage to use
        
    Returns:
        ESTformer model, sigmas module
    """
    # Create the model
    model = ESTFormer(
        lr_channel_names=lr_channel_names,
        hr_channel_names=hr_channel_names,
        builtin_montage=builtin_montage,
        time_steps=time_steps,
        alpha_s=alpha_s,
        alpha_t=alpha_t,
        r_mlp=r_mlp,
        dropout_rate=dropout_rate,
        L_s=L_s,
        L_t=L_t
    )
    
    # Create trainable sigma parameters for the loss function
    sigmas = SigmaParameters()
    
    return model, sigmas

def get_data_loaders(lr_channel_names, hr_channel_names, batch_size=36, dataset_split="70/25/5", random_state=97, eeg_epoch_mode="fixed_length_event", fixed_length_duration=6, duration_before_onset=0.05, duration_after_onset=0.6):
    """
    Create data loaders for training and validation.
    
    Args:
        lr_channel_names: List of low-resolution channel names
        hr_channel_names: List of high-resolution channel names
        batch_size: Batch size
        dataset_split: Fraction of data to use for validation
        fixed_length_duration: Duration of each event in seconds
        
    Returns:
        Training and validation data loaders
    """
    train_dataset = HDF5DataSplitGenerator(
        dataset_type="train",
        dataset_split=dataset_split,
        eeg_epoch_mode=eeg_epoch_mode,
        random_state=random_state,
        fixed_length_duration=fixed_length_duration,
        duration_before_onset=duration_before_onset,
        duration_after_onset=duration_after_onset,
        lr_channel_names=lr_channel_names,
        hr_channel_names=hr_channel_names
    )
    
    val_dataset = HDF5DataSplitGenerator(
        dataset_type="val",
        dataset_split=dataset_split,
        eeg_epoch_mode=eeg_epoch_mode,
        random_state=random_state,
        fixed_length_duration=fixed_length_duration,
        duration_before_onset=duration_before_onset,
        duration_after_onset=duration_after_onset,
        lr_channel_names=lr_channel_names,
        hr_channel_names=hr_channel_names
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def train_estformer(model, train_loader, val_loader, sigmas, device, eeg_epoch_mode="fixed_length_event", fixed_length_duration=8, lr=5e-5, beta_1=0.9, beta_2=0.95, epochs=30, weight_decay=0.5, checkpoint_dir='checkpoints'):
    """
    Train the ESTformer model.
    
    Args:
        model: ESTformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        sigmas: SigmaParameters module
        device: Device to train on (cuda/cpu)
        lr: Learning rate
        epochs: Number of epochs to train for
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize wandb
    config = {
        "model": "ESTformer",
        "eeg_epoch_mode": eeg_epoch_mode,
        "fixed_length_duration": fixed_length_duration,
        "lr_channels": len(train_loader.dataset.lr_indices) if train_loader.dataset.lr_indices is not None else len(train_loader.dataset.ch_names),
        "hr_channels": len(train_loader.dataset.hr_indices) if train_loader.dataset.hr_indices is not None else len(train_loader.dataset.ch_names),
        "alpha_s": model.alpha_s,
        "alpha_t": model.alpha_t,
        "r_mlp": model.sim.cab1.blocks[0]['tsab1'].layers[0]['mlp'][0].out_features,
        "dropout_rate": model.sim.cab1.blocks[0]['tsab1'].layers[0]['dropout1'].p,
        "L_s": len(model.sim.cab1.blocks),
        "L_t": len(model.trm.tsab1.layers),
        "batch_size": train_loader.batch_size,
        "epochs": epochs,
        "optimizer": "Adam",
        "learning_rate": lr,
    }
    
    wandb.init(project="eeg-estformer", config=config)
    
    # Create optimizer with both model and sigma parameters
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': sigmas.parameters()}
    ], lr=lr, weight_decay=weight_decay, betas=(beta_1, beta_2))
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'sigma1': [],
        'sigma2': []
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        sigmas.train()
        train_losses = []
        train_maes = []
        train_nmse = []
        train_snr = []
        train_pcc = []
        
        # Training phase
        # train_dataset = train_loader.dataset
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for i, batch in enumerate(progress_bar):
            lo_res = batch['lo_res'].float().to(device)
            hi_res = batch['hi_res'].float().to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(lo_res)

            # Compute loss
            loss = reconstruction_loss(hi_res, outputs, sigmas.sigma1, sigmas.sigma2)
            mae = compute_mean_absolute_error(hi_res, outputs)
            nmse = compute_normalized_mean_squared_error(hi_res, outputs)
            snr = compute_signal_to_noise_ratio(hi_res, outputs)
            pcc = compute_pearson_correlation_coefficient(hi_res, outputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track metrics
            train_losses.append(loss.item())
            train_maes.append(mae.item())
            train_nmse.append(nmse.item())
            train_snr.append(snr.item())
            train_pcc.append(pcc.item())


            # Update tqdm bar with metrics
            if i % 10 == 0:
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", mae=f"{mae.item():.4f}", nmse=f"{nmse.item():.4f}", snr=f"{snr.item():.4f}", pcc=f"{pcc.item():.4f}")

            # Free memory
            del lo_res, hi_res, outputs, loss, mae, nmse, snr, pcc
            torch.cuda.empty_cache()
        
        # Compute epoch metrics
        avg_train_loss = np.mean(train_losses)
        avg_train_mae = np.mean(train_maes)
        avg_train_nmse = np.mean(train_nmse)
        avg_train_snr = np.mean(train_snr)
        avg_train_pcc = np.mean(train_pcc)
        
        # Validation phase
        model.eval()
        sigmas.eval()
        val_losses = []
        val_maes = []
        val_nmse = []
        val_snr = []
        val_pcc = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Passing validation samples", leave=False)
            for batch in val_loader:
                lo_res = batch['lo_res'].float().to(device)
                hi_res = batch['hi_res'].float().to(device)
                
                # Forward pass
                outputs = model(lo_res)
                
                # Compute loss
                loss = reconstruction_loss(hi_res, outputs, sigmas.sigma1, sigmas.sigma2)
                mae = compute_mean_absolute_error(hi_res, outputs)
                nmse = compute_normalized_mean_squared_error(hi_res, outputs)
                snr = compute_signal_to_noise_ratio(hi_res, outputs)
                pcc = compute_pearson_correlation_coefficient(hi_res, outputs)
                
                # Track metrics
                val_losses.append(loss.item())
                val_maes.append(mae.item())
                val_nmse.append(nmse.item())
                val_snr.append(snr.item())
                val_pcc.append(pcc.item())

                # Free memory
                del lo_res, hi_res, outputs, loss, mae
                torch.cuda.empty_cache()
        
        # Compute epoch metrics
        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        avg_val_nmse = np.mean(val_nmse)
        avg_val_snr = np.mean(val_snr)
        avg_val_pcc = np.mean(val_pcc)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mae'].append(avg_train_mae)
        history['val_mae'].append(avg_val_mae)
        history['train_nmse'].append(avg_train_nmse)
        history['val_nmse'].append(avg_val_nmse)
        history['train_snr'].append(avg_train_snr)
        history['val_snr'].append(avg_val_snr)
        history['train_pcc'].append(avg_train_pcc)
        history['val_pcc'].append(avg_val_pcc)
        history['sigma1'].append(sigmas.sigma1.item())
        history['sigma2'].append(sigmas.sigma2.item())
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_mae": avg_train_mae,
            "val_mae": avg_val_mae,
            "train_nmse": avg_train_nmse,
            "val_nmse": avg_val_nmse,
            "train_snr": avg_train_snr,
            "val_snr": avg_val_snr,
            "train_pcc": avg_train_pcc,
            "val_pcc": avg_val_pcc,
            "sigma1": sigmas.sigma1.item(),
            "sigma2": sigmas.sigma2.item()
        })
        
        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"train_loss: {avg_train_loss:.4f}, "
            f"val_loss: {avg_val_loss:.4f}, "
            f"train_mae: {avg_train_mae:.4f}, "
            f"val_mae: {avg_val_mae:.4f}, "
            f"train_nmse: {avg_train_nmse:.4f}, "
            f"val_nmse: {avg_val_nmse:.4f}, "
            f"train_snr: {avg_train_snr:.4f}, "
            f"val_snr: {avg_val_snr:.4f}, "
            f"train_pcc: {avg_train_pcc:.4f}, "
            f"val_pcc: {avg_val_pcc:.4f}, "
            f"sigma1: {sigmas.sigma1.item():.4f}, "
            f"sigma2: {sigmas.sigma2.item():.4f}"
        )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'estformer_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'sigmas_state_dict': sigmas.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            
            print(f"Saved best model checkpoint to {checkpoint_path}")
            # Log best model to wandb
            wandb.save(checkpoint_path, policy='now')
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'estformer_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'sigmas_state_dict': sigmas.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
    
    # Close wandb run
    wandb.finish()
    
    return history

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
#     plt.title(f'Low-Res Input (Channel {channel_idx})')
    
#     # Plot high-res ground truth
#     plt.subplot(3, 1, 2)
#     plt.plot(hi_res[channel_idx])
#     plt.title(f'High-Res Ground Truth (Channel {channel_idx})')
    
#     # Plot prediction
#     plt.subplot(3, 1, 3)
#     plt.plot(pred[channel_idx])
#     plt.title(f'Model Prediction (Channel {channel_idx})')
    
#     plt.tight_layout()
    
#     # Save figure to wandb
#     if wandb.run is not None:
#         wandb.log({"prediction_visualization": wandb.Image(plt)})
    
#     plt.show()

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

def main():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Print available GPU memory
    if torch.cuda.is_available():
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Define parameters
    all_channels = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

    # Model parameters
    lr_channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'] # Low-resolution setup (fewer channels)
    hr_channel_names = all_channels # High-resolution setup (all channels)
    eeg_epoch_mode = "fixed_length"
    fixed_length_duration = 6
    alpha_t = 0.60
    alpha_s = 0.75
    r_mlp = 4 # amplification factor for MLP layers
    dropout_rate = 0.5
    L_s = 1  # Number of spatial layers
    L_t = 1  # Number of temporal layers

    # Training parameters
    batch_size = 30
    epochs = 30

    # Optimizer parameters
    lr = 5e-5
    weight_decay = 0.5
    beta_1 = 0.9
    beta_2 = 0.95

    # Dataset parameters
    dataset_split = "70/25/5"
    
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        lr_channel_names=lr_channel_names,
        hr_channel_names=hr_channel_names,
        batch_size=batch_size,
        dataset_split=dataset_split,
        eeg_epoch_mode=eeg_epoch_mode,
        fixed_length_duration=fixed_length_duration
    )
    
    # Get sample data to determine time_steps
    sample_item = train_loader.dataset[0]
    time_steps = sample_item["lo_res"].shape[1]
    
    # Create the model
    model, sigmas = create_estformer_model(
        lr_channel_names=lr_channel_names,
        hr_channel_names=hr_channel_names,
        time_steps=time_steps,
        alpha_s=alpha_s,
        alpha_t=alpha_t,
        r_mlp=r_mlp,
        dropout_rate=dropout_rate,
        L_s=L_s,
        L_t=L_t
    )
    
    # Move model to device
    model = model.to(device)
    sigmas = sigmas.to(device)
    
    # Print model summary
    # print(model)
    # print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    history = train_estformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        sigmas=sigmas,
        device=device,
        eeg_epoch_mode=eeg_epoch_mode,
        fixed_length_duration=fixed_length_duration,
        lr=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        epochs=epochs,
        weight_decay=weight_decay,
        checkpoint_dir='checkpoints'
    )
    
    # Monitor sigma values and loss
    # monitor_sigma_values_and_loss(history)
    
    # Visualize results
    # visualize_results(model, val_loader.dataset, device)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()