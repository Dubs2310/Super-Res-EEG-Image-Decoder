import torch
import wandb
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from mne.channels import make_standard_montage
from torchmetrics.aggregation import MeanMetric
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from models.definers.super_resolution import EEGSuperResolutionDefiner as EEGSuperResolution

class SuperResolutionTrainerModel(pl.LightningModule):
    def __init__(self, lo_res_channel_names, hi_res_channel_names, time_steps, lr=5e-5, weight_decay=0.5, beta1=0.9, beta2=0.95, builtin_montage='standard_1020'):
        super().__init__()
        self.super_resolution_model = EEGSuperResolution(lo_res_channel_names, hi_res_channel_names, time_steps)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.lo_res_channel_names = lo_res_channel_names
        self.hi_res_channel_names = hi_res_channel_names
        self.builtin_montage = builtin_montage
        self.save_hyperparameters(ignore=['model'])
        
        # Training and validation metrics
        self.mean_train_loss = MeanMetric()
        self.mean_train_mae = MeanAbsoluteError()
        self.mean_val_loss = MeanMetric()
        self.mean_val_mae = MeanAbsoluteError()
        
        # Test metrics
        self.mean_test_loss = MeanMetric()
        self.mean_test_snr = SignalNoiseRatio()
        self.mean_test_mae = MeanAbsoluteError()
        self.mean_test_mse = MeanSquaredError()
        
        # Manual NMSE calculation (simpler than NRMSE)
        self.manual_mse_sum = 0.0
        self.manual_target_var_sum = 0.0
        self.manual_target_mean_sum = 0.0
        self.manual_n_samples = 0
        
        # For Pearson correlation, we'll compute it manually to avoid dimension issues
        self.pearson_sum_xy = 0.0
        self.pearson_sum_x = 0.0
        self.pearson_sum_y = 0.0
        self.pearson_sum_x2 = 0.0
        self.pearson_sum_y2 = 0.0
        self.pearson_n = 0

    def training_step(self, batch, batch_idx):
        lo_res, hi_res = batch
        lo_res = (lo_res[0] if len(lo_res) == 2 else lo_res).float()
        hi_res = (hi_res[0] if len(hi_res) == 2 else hi_res).float()
        super_res = self.super_resolution_model(lo_res)
        loss = self.super_resolution_model.compute_loss(hi_res, super_res)
        
        # Update metrics - flatten for MAE if needed
        self.mean_train_loss(loss)
        self.mean_train_mae(super_res, hi_res)
        
        self.log("train_batch_loss", self.mean_train_loss)
        return loss

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def on_train_epoch_end(self):
        self.log("train_loss", self.mean_train_loss, prog_bar=True)
        self.log("train_mae", self.mean_train_mae, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        lo_res, hi_res = batch
        lo_res = (lo_res[0] if len(lo_res) == 2 else lo_res).float()
        hi_res = (hi_res[0] if len(hi_res) == 2 else hi_res).float()
        super_res = self.super_resolution_model(lo_res)
        loss = self.super_resolution_model.compute_loss(hi_res, super_res)
        
        # Update metrics
        self.mean_val_loss(loss)
        self.mean_val_mae(super_res, hi_res)
        
        return loss

    def on_validation_epoch_end(self):
        self.log("val_loss", self.mean_val_loss, prog_bar=True)
        self.log("val_mae", self.mean_val_mae, prog_bar=True)

    def test_step(self, batch, batch_idx):
        lo_res, hi_res = batch
        lo_res = (lo_res[0] if len(lo_res) == 2 else lo_res).float()
        hi_res = (hi_res[0] if len(hi_res) == 2 else hi_res).float()
        super_res = self.super_resolution_model(lo_res)
        loss = self.super_resolution_model.compute_loss(hi_res, super_res)
        
        # Update all test metrics
        self.mean_test_loss(loss)
        self.mean_test_snr(super_res, hi_res)
        self.mean_test_mae(super_res, hi_res)
        self.mean_test_mse(super_res, hi_res)
        
        # Manual NMSE calculation
        with torch.no_grad():
            # Calculate MSE for this batch
            mse_batch = torch.mean((super_res - hi_res) ** 2)
            
            # Calculate normalization factors
            target_var = torch.var(hi_res)  # Variance normalization (most common for NMSE)
            target_mean_sq = torch.mean(hi_res ** 2)  # Mean square normalization
            
            # Accumulate for manual calculation
            self.manual_mse_sum += mse_batch.cpu().item()
            self.manual_target_var_sum += target_var.cpu().item()
            self.manual_target_mean_sum += target_mean_sq.cpu().item()
            self.manual_n_samples += 1
        
        # Flatten tensors for Pearson correlation and compute manually
        super_res_flat = super_res.view(-1).detach().cpu()  # Complete flattening
        hi_res_flat = hi_res.view(-1).detach().cpu()  # Complete flattening
        
        # Update Pearson correlation statistics manually
        self.pearson_sum_xy += torch.sum(super_res_flat * hi_res_flat).item()
        self.pearson_sum_x += torch.sum(super_res_flat).item()
        self.pearson_sum_y += torch.sum(hi_res_flat).item()
        self.pearson_sum_x2 += torch.sum(super_res_flat ** 2).item()
        self.pearson_sum_y2 += torch.sum(hi_res_flat ** 2).item()
        self.pearson_n += super_res_flat.numel()
        
        return loss

    def on_test_epoch_end(self):
        # Log all test metrics
        test_loss = self.mean_test_loss.compute()
        test_snr = self.mean_test_snr.compute()
        test_mae = self.mean_test_mae.compute()
        test_mse = self.mean_test_mse.compute()
        
        # Compute Pearson correlation manually
        if self.pearson_n > 1:
            numerator = self.pearson_n * self.pearson_sum_xy - self.pearson_sum_x * self.pearson_sum_y
            denominator_x = self.pearson_n * self.pearson_sum_x2 - self.pearson_sum_x ** 2
            denominator_y = self.pearson_n * self.pearson_sum_y2 - self.pearson_sum_y ** 2
            
            if denominator_x > 0 and denominator_y > 0:
                test_pearson = numerator / (denominator_x * denominator_y) ** 0.5
            else:
                test_pearson = 0.0
        else:
            test_pearson = 0.0
        
        # Compute manual NMSE
        if self.manual_n_samples > 0:
            avg_mse = self.manual_mse_sum / self.manual_n_samples
            avg_target_var = self.manual_target_var_sum / self.manual_n_samples
            avg_target_mean_sq = self.manual_target_mean_sum / self.manual_n_samples
            
            # Two common NMSE formulations
            nmse_var = avg_mse / avg_target_var if avg_target_var > 0 else float('inf')
            nmse_mean_sq = avg_mse / avg_target_mean_sq if avg_target_mean_sq > 0 else float('inf')
            
            # Use variance-based NMSE (most common)
            test_nmse = nmse_var
            
            print(f"\n=== NMSE ANALYSIS ===")
            print(f"NMSE (variance): {nmse_var:.6f}")
            print(f"NMSE (mean square): {nmse_mean_sq:.6f}")
            print(f"Average MSE: {avg_mse:.6f}")
            print(f"Average Target Variance: {avg_target_var:.6f}")
            print(f"Average Target Mean Square: {avg_target_mean_sq:.6f}")
            print(f"===================\n")
        else:
            test_nmse = 0.0
        
        self.log("test_loss", test_loss)
        self.log("test_snr", test_snr)
        self.log("test_mae", test_mae)
        self.log("test_mse", test_mse)
        self.log("test_nmse", test_nmse)
        self.log("test_pearson", test_pearson)
        
        # Print test results
        print(f"\n=== SUPER RESOLUTION TEST RESULTS ===")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Test SNR: {test_snr:.4f} dB")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Test NMSE: {test_nmse:.6f}")
        print(f"Test Pearson Correlation: {test_pearson:.4f}")
        print(f"=====================================\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))
        return optimizer


class SuperResolutionPlottingCallback(pl.Callback):
    def __init__(self, builtin_montage='standard_1020'):
        # Training curves data
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        
        # Test metrics data
        self.test_metrics = {}
        
        # For EEG epoch visualization
        self.eeg_samples = []
        self.best_sample_idx = None
        self.best_sample_metrics = None
        
        # Store montage for channel analysis
        self.builtin_montage = builtin_montage
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Collect training metrics
        train_loss = trainer.callback_metrics.get('train_loss', 0)
        train_mae = trainer.callback_metrics.get('train_mae', 0)
        
        # Convert tensors to Python floats
        if torch.is_tensor(train_loss):
            train_loss = train_loss.cpu().item()
        if torch.is_tensor(train_mae):
            train_mae = train_mae.cpu().item()
            
        self.train_losses.append(train_loss)
        self.train_maes.append(train_mae)
        
        # Debug print
        print(f"Epoch {len(self.train_losses)}: Train Loss = {train_loss:.6f}, Train MAE = {train_mae:.6f}")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect validation metrics
        val_loss = trainer.callback_metrics.get('val_loss', 0)
        val_mae = trainer.callback_metrics.get('val_mae', 0)
        
        # Convert tensors to Python floats
        if torch.is_tensor(val_loss):
            val_loss = val_loss.cpu().item()
        if torch.is_tensor(val_mae):
            val_mae = val_mae.cpu().item()
            
        self.val_losses.append(val_loss)
        self.val_maes.append(val_mae)
        
        # Debug print
        print(f"Epoch {len(self.val_losses)}: Val Loss = {val_loss:.6f}, Val MAE = {val_mae:.6f}")
        print(f"Data lengths: Train={len(self.train_losses)}, Val={len(self.val_losses)}")
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect EEG samples for visualization"""
        lo_res, hi_res = batch
        lo_res = (lo_res[0] if len(lo_res) == 2 else lo_res).float()
        hi_res = (hi_res[0] if len(hi_res) == 2 else hi_res).float()
        
        with torch.no_grad():
            super_res = pl_module.super_resolution_model(lo_res)
            
            # Calculate per-sample metrics to find the best one
            batch_size = hi_res.size(0)
            for i in range(min(batch_size, 5)):  # Limit to 5 samples per batch for memory
                sample_hi_res = hi_res[i]
                sample_super_res = super_res[i]
                sample_lo_res = lo_res[i]
                
                # Calculate sample-specific metrics
                mse = torch.mean((sample_super_res - sample_hi_res) ** 2).item()
                mae = torch.mean(torch.abs(sample_super_res - sample_hi_res)).item()
                
                # Calculate correlation for this sample
                flat_super = sample_super_res.view(-1).cpu()
                flat_hi = sample_hi_res.view(-1).cpu()
                
                if len(flat_super) > 1 and torch.std(flat_super) > 1e-8 and torch.std(flat_hi) > 1e-8:
                    correlation = torch.corrcoef(torch.stack([flat_super, flat_hi]))[0, 1].item()
                else:
                    correlation = 0.0
                
                # Store sample data with channel information
                sample_data = {
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'global_idx': batch_idx * batch_size + i,
                    'lo_res': sample_lo_res.cpu().numpy(),
                    'hi_res': sample_hi_res.cpu().numpy(),
                    'super_res': sample_super_res.cpu().numpy(),
                    'mse': mse,
                    'mae': mae,
                    'correlation': correlation,
                    'combined_score': correlation - mse,  # Higher correlation, lower MSE is better
                    'lo_res_channels': pl_module.lo_res_channel_names,
                    'hi_res_channels': pl_module.hi_res_channel_names
                }
                
                self.eeg_samples.append(sample_data)
        
        # Keep only the best 50 samples to manage memory
        if len(self.eeg_samples) > 50:
            self.eeg_samples.sort(key=lambda x: x['combined_score'], reverse=True)
            self.eeg_samples = self.eeg_samples[:50]
    
    def on_test_epoch_end(self, trainer, pl_module):
        # Wait for the model to compute its metrics first
        # The metrics should be available in trainer.callback_metrics after model's on_test_epoch_end
        
        # Give the model a chance to compute metrics
        if hasattr(pl_module, 'on_test_epoch_end'):
            # Access the computed values directly from the model
            test_loss = pl_module.mean_test_loss.compute()
            test_snr = pl_module.mean_test_snr.compute() 
            test_mae = pl_module.mean_test_mae.compute()
            test_mse = pl_module.mean_test_mse.compute()
            test_nmse = pl_module.manual_mse_sum / pl_module.manual_n_samples / (pl_module.manual_target_var_sum / pl_module.manual_n_samples) if pl_module.manual_n_samples > 0 and pl_module.manual_target_var_sum > 0 else 0.0
            
            # Get Pearson correlation from model's manual calculation
            if hasattr(pl_module, 'pearson_n') and pl_module.pearson_n > 1:
                numerator = pl_module.pearson_n * pl_module.pearson_sum_xy - pl_module.pearson_sum_x * pl_module.pearson_sum_y
                denominator_x = pl_module.pearson_n * pl_module.pearson_sum_x2 - pl_module.pearson_sum_x ** 2
                denominator_y = pl_module.pearson_n * pl_module.pearson_sum_y2 - pl_module.pearson_sum_y ** 2
                
                if denominator_x > 0 and denominator_y > 0:
                    test_pearson = numerator / (denominator_x * denominator_y) ** 0.5
                else:
                    test_pearson = 0.0
            else:
                test_pearson = 0.0
            
            # Convert tensors to Python floats
            test_metrics = {
                'test_loss': test_loss.cpu().item() if torch.is_tensor(test_loss) else test_loss,
                'test_snr': test_snr.cpu().item() if torch.is_tensor(test_snr) else test_snr,
                'test_mae': test_mae.cpu().item() if torch.is_tensor(test_mae) else test_mae,
                'test_mse': test_mse.cpu().item() if torch.is_tensor(test_mse) else test_mse,
                'test_nmse': test_nmse,
                'test_pearson': test_pearson
            }
            
            # Store the metrics
            self.test_metrics = test_metrics
            
            print(f"Callback collected test metrics: {self.test_metrics}")
        
        # Create test metrics bar plot
        self._create_test_metrics_plot(trainer)
    
    def _get_missing_channel_indices(self, lo_res_channels, hi_res_channels):
        """Get indices of channels that are missing from low-res but present in high-res"""
        try:
            # Get montage positions
            montage = make_standard_montage(self.builtin_montage)
            pos_dict = montage.get_positions()['ch_pos']
            
            # Find missing channels (in hi_res but not in lo_res)
            missing_channels = [ch for ch in hi_res_channels if ch not in lo_res_channels]
            
            # Get indices of missing channels in the hi_res array
            missing_indices = []
            for ch in missing_channels:
                if ch in hi_res_channels:
                    missing_indices.append(hi_res_channels.index(ch))
            
            print(f"Missing channels (being super-resolved): {missing_channels}")
            print(f"Missing channel indices: {missing_indices}")
            
            return missing_indices, missing_channels
            
        except Exception as e:
            print(f"Error getting channel indices: {e}")
            # Fallback: assume first few channels are missing
            n_missing = len(hi_res_channels) - len(lo_res_channels)
            missing_indices = list(range(min(n_missing, 8)))
            missing_channels = hi_res_channels[:len(missing_indices)]
            return missing_indices, missing_channels
    
    def _get_available_channel_indices(self, lo_res_channels, hi_res_channels):
        """Get indices of channels that are available in both low-res and high-res"""
        try:
            # Find common channels (should be identical since low-res is just dropped channels)
            common_channels = [ch for ch in lo_res_channels if ch in hi_res_channels]
            
            # Get indices in hi_res array - maintaining order from lo_res_channels
            common_hi_indices = []
            lo_res_indices = []
            
            for i, ch in enumerate(lo_res_channels):
                if ch in hi_res_channels:
                    hi_idx = hi_res_channels.index(ch)
                    common_hi_indices.append(hi_idx)
                    lo_res_indices.append(i)  # Use position in lo_res array
            
            print(f"Available channels: {common_channels}")
            print(f"Lo-res channel list: {lo_res_channels}")
            print(f"Hi-res channel list: {hi_res_channels}")
            print(f"Available indices in hi_res: {common_hi_indices}")
            print(f"Available indices in lo_res: {lo_res_indices}")
            
            # Debug: Check if indices match expected channels
            for i, (lo_idx, hi_idx, ch) in enumerate(zip(lo_res_indices, common_hi_indices, common_channels)):
                expected_lo_ch = lo_res_channels[lo_idx] if lo_idx < len(lo_res_channels) else "INDEX_ERROR"
                expected_hi_ch = hi_res_channels[hi_idx] if hi_idx < len(hi_res_channels) else "INDEX_ERROR"
                print(f"  {ch}: lo_idx={lo_idx} ({expected_lo_ch}), hi_idx={hi_idx} ({expected_hi_ch})")
            
            return common_hi_indices, lo_res_indices, common_channels
            
        except Exception as e:
            print(f"Error getting common channel indices: {e}")
            # Fallback
            n_common = min(len(lo_res_channels), len(hi_res_channels), 4)
            common_hi_indices = list(range(n_common))
            lo_res_indices = list(range(n_common))
            common_channels = lo_res_channels[:n_common]
            return common_hi_indices, lo_res_indices, common_channels
    
    def _create_training_curves_plot(self, trainer):
        """Create separate plots for loss and MAE training curves"""
        
        # Loss curves plot
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log to wandb if available
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"training_loss_curves": wandb.Image(plt)})
        
        plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # MAE curves plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_maes, 'b-', label='Training MAE', linewidth=2)
        plt.plot(epochs, self.val_maes, 'r-', label='Validation MAE', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training vs Validation MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log to wandb if available
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"training_mae_curves": wandb.Image(plt)})
        
        plt.savefig('training_mae_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_test_metrics_plot(self, trainer):
        """Create bar plot for test metrics"""
        if not self.test_metrics:
            print("No test metrics available for plotting")
            return
        
        # Debug: Print the actual values we're about to plot
        print(f"Plotting test metrics: {self.test_metrics}")
        
        # Check if all values are zero (which shouldn't happen)
        if all(abs(v) < 1e-10 for v in self.test_metrics.values()):
            print("Warning: All test metrics are zero or very close to zero")
            return
        
        # Prepare data for plotting
        metric_names = list(self.test_metrics.keys())
        metric_values = list(self.test_metrics.values())
        
        # Create more readable labels
        display_names = {
            'test_loss': 'Loss',
            'test_mae': 'MAE',
            'test_mse': 'MSE', 
            'test_nmse': 'NMSE',
            'test_pearson': 'Pearson Corr.',
            'test_snr': 'SNR (dB)'
        }
        
        # Create the bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bars with different colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        bars = ax.bar(range(len(metric_names)), metric_values, 
                     color=colors[:len(metric_names)], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the plot
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_title('Super Resolution Test Metrics', fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis labels
        display_labels = [display_names.get(name, name) for name in metric_names]
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(display_labels, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            height = bar.get_height()
            if metric_names[i] == 'test_pearson':
                # Format correlation values differently
                ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01 if height >= 0 else height - abs(height)*0.01,
                       f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
            elif metric_names[i] == 'test_snr':
                # Format SNR values
                ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01 if height >= 0 else height - abs(height)*0.01,
                       f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
            else:
                # Format other metrics in scientific notation if very small
                if abs(value) < 0.001 and value != 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01 if height >= 0 else height - abs(height)*0.01,
                           f'{value:.2e}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01 if height >= 0 else height - abs(height)*0.01,
                           f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Log to wandb if available
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"test_metrics_bar_plot": wandb.Image(fig)})
            print("Logged test metrics bar plot to wandb")
        
        plt.savefig('test_metrics_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a table-style summary
        self._create_test_metrics_summary_plot(trainer)
    
    def _create_test_metrics_summary_plot(self, trainer):
        """Create a summary table plot of test metrics"""
        if not self.test_metrics:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # Prepare data for table
        metric_info = {
            'test_loss': ('Loss', f"{self.test_metrics['test_loss']:.6f}", 'Lower is better'),
            'test_mae': ('Mean Absolute Error', f"{self.test_metrics['test_mae']:.6f}", 'Lower is better'),
            'test_mse': ('Mean Squared Error', f"{self.test_metrics['test_mse']:.6f}", 'Lower is better'),
            'test_nmse': ('Normalized MSE', f"{self.test_metrics['test_nmse']:.6f}", 'Lower is better'),
            'test_pearson': ('Pearson Correlation', f"{self.test_metrics['test_pearson']:.4f}", 'Higher is better'),
            'test_snr': ('Signal-to-Noise Ratio', f"{self.test_metrics['test_snr']:.2f} dB", 'Higher is better')
        }
        
        # Create table data
        table_data = []
        for metric_key in self.test_metrics.keys():
            if metric_key in metric_info:
                name, value, interpretation = metric_info[metric_key]
                table_data.append([name, value, interpretation])
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value', 'Interpretation'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Color header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('Super Resolution Test Metrics Summary', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Log to wandb if available
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"test_metrics_summary": wandb.Image(fig)})
        
        plt.savefig('test_metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_best_eeg_epoch_plot(self, trainer):
        """Create visualization of the best EEG epoch showing super-resolved missing channels"""
        if not self.eeg_samples:
            print("No EEG samples available for plotting")
            return
        
        # Find the best sample (highest combined score)
        best_sample = max(self.eeg_samples, key=lambda x: x['combined_score'])
        
        print(f"Creating EEG epoch plot for sample {best_sample['global_idx']} "
              f"(correlation: {best_sample['correlation']:.4f}, MSE: {best_sample['mse']:.6f})")
        
        # Get the data and channel information
        lo_res = best_sample['lo_res']  # Shape: (lo_res_channels, time_steps)
        hi_res = best_sample['hi_res']  # Shape: (hi_res_channels, time_steps)
        super_res = best_sample['super_res']  # Shape: (hi_res_channels, time_steps)
        lo_res_channels = best_sample['lo_res_channels']
        hi_res_channels = best_sample['hi_res_channels']
        
        # Debug the data alignment issue
        print(f"\n=== DATA ALIGNMENT DEBUG ===")
        print(f"Lo-res shape: {lo_res.shape}, channels: {lo_res_channels}")
        print(f"Hi-res shape: {hi_res.shape}, channels: {hi_res_channels}")
        print(f"Super-res shape: {super_res.shape}")
        
        # Get missing channels (channels that are super-resolved)
        missing_indices, missing_channels = self._get_missing_channel_indices(lo_res_channels, hi_res_channels)
        
        # Get common channels (for reference) - these should be IDENTICAL
        common_hi_indices, common_lo_indices, common_channels = self._get_available_channel_indices(lo_res_channels, hi_res_channels)
        
        # Check if available channels are actually identical
        print(f"\n=== AVAILABLE CHANNEL VERIFICATION ===")
        for i, (hi_idx, lo_idx, ch_name) in enumerate(zip(common_hi_indices[:3], common_lo_indices[:3], common_channels[:3])):
            if i < len(common_channels):
                lo_data = lo_res[lo_idx]
                hi_data = hi_res[hi_idx]
                super_data = super_res[hi_idx]
                
                # Check if lo_res and hi_res are identical (they should be!)
                mse_lo_hi = np.mean((lo_data - hi_data) ** 2)
                correlation_lo_hi = np.corrcoef(lo_data, hi_data)[0, 1] if np.std(lo_data) > 1e-8 and np.std(hi_data) > 1e-8 else np.nan
                
                # Check how much super-res differs from ground truth
                mse_super_hi = np.mean((super_data - hi_data) ** 2)
                correlation_super_hi = np.corrcoef(super_data, hi_data)[0, 1] if np.std(super_data) > 1e-8 and np.std(hi_data) > 1e-8 else np.nan
                
                print(f"{ch_name} (lo_idx={lo_idx}, hi_idx={hi_idx}):")
                print(f"  Lo-res vs Hi-res (should be identical): MSE={mse_lo_hi:.8f}, Corr={correlation_lo_hi:.6f}")
                print(f"  Super-res vs Hi-res: MSE={mse_super_hi:.8f}, Corr={correlation_super_hi:.6f}")
                print(f"  Lo-res range: [{np.min(lo_data):.4f}, {np.max(lo_data):.4f}]")
                print(f"  Hi-res range: [{np.min(hi_data):.4f}, {np.max(hi_data):.4f}]")
        
        # Limit number of channels to plot for readability
        max_missing_to_plot = min(len(missing_indices), 6)
        max_common_to_plot = min(len(common_hi_indices), 4)
        
        missing_indices_to_plot = missing_indices[:max_missing_to_plot]
        missing_channels_to_plot = missing_channels[:max_missing_to_plot]
        
        common_hi_indices_to_plot = common_hi_indices[:max_common_to_plot]
        common_lo_indices_to_plot = common_lo_indices[:max_common_to_plot]
        common_channels_to_plot = common_channels[:max_common_to_plot]
        
        # Create subplots: missing channels + some common channels for reference
        total_plots = len(missing_indices_to_plot) + len(common_hi_indices_to_plot)
        
        if total_plots == 0:
            print("No channels to plot")
            return
        
        fig, axes = plt.subplots(total_plots, 1, figsize=(15, 2*total_plots), sharex=True)
        if total_plots == 1:
            axes = [axes]
        
        # Time axes - should be identical for available channels
        time_lo = np.linspace(0, 1, lo_res.shape[1])
        time_hi = np.linspace(0, 1, hi_res.shape[1])
        
        plot_idx = 0
        
        # Plot missing channels (super-resolved)
        for i, (ch_idx, ch_name) in enumerate(zip(missing_indices_to_plot, missing_channels_to_plot)):
            ax = axes[plot_idx]
            
            # Plot ground truth and super-resolution for missing channels
            ax.plot(time_hi, hi_res[ch_idx], 'k-', linewidth=1.5, label='Ground Truth (High-res)', alpha=0.8)
            ax.plot(time_hi, super_res[ch_idx], 'r-', linewidth=1.2, label='Super Resolution', alpha=0.9)
            
            # Customize subplot
            ax.set_ylabel(f'{ch_name}\n(Missing)\nAmplitude', fontsize=10, color='red')
            ax.grid(True, alpha=0.3)
            
            # Add legend only to the first subplot
            if plot_idx == 0:
                ax.legend(loc='upper right', fontsize=9)
            
            # Set y-axis limits
            all_values = np.concatenate([hi_res[ch_idx], super_res[ch_idx]])
            y_margin = np.std(all_values) * 0.1 if np.std(all_values) > 0 else 0.1
            ax.set_ylim(np.min(all_values) - y_margin, np.max(all_values) + y_margin)
            
            plot_idx += 1
        
        # Plot some common channels for reference - these should show lo_res = hi_res
        for i, (hi_idx, lo_idx, ch_name) in enumerate(zip(common_hi_indices_to_plot, common_lo_indices_to_plot, common_channels_to_plot)):
            ax = axes[plot_idx]
            
            # Plot all three signals for common channels
            ax.plot(time_hi, hi_res[hi_idx], 'k-', linewidth=2, label='Ground Truth (Hi-res)', alpha=0.8)
            ax.plot(time_hi, super_res[hi_idx], 'r-', linewidth=1.2, label='Super Resolution', alpha=0.9)
            ax.plot(time_lo, lo_res[lo_idx], 'bo-', markersize=4, linewidth=1.5, label='Low Resolution Input', alpha=0.9)
            
            # Calculate and display the difference
            if lo_res.shape[1] == hi_res.shape[1]:
                mse_diff = np.mean((lo_res[lo_idx] - hi_res[hi_idx]) ** 2)
                ax.set_title(f'Lo-res vs Hi-res MSE: {mse_diff:.8f}', fontsize=9)
            
            # Customize subplot
            ax.set_ylabel(f'{ch_name}\n(Available)\nAmplitude', fontsize=10, color='blue')
            ax.grid(True, alpha=0.3)
            
            # Add legend for reference channels
            if plot_idx == len(missing_indices_to_plot):  # First reference channel
                ax.legend(loc='upper right', fontsize=9)
            
            # Set y-axis limits
            all_values = np.concatenate([hi_res[hi_idx], super_res[hi_idx], lo_res[lo_idx]])
            y_margin = np.std(all_values) * 0.1 if np.std(all_values) > 0 else 0.1
            ax.set_ylim(np.min(all_values) - y_margin, np.max(all_values) + y_margin)
            
            plot_idx += 1
        
        # Set common x-axis label
        axes[-1].set_xlabel('Normalized Time', fontsize=12)
        
        # Add overall title with metrics and channel info
        fig.suptitle(f'EEG Super-Resolution: Missing Channels Reconstruction\n'
                    f'Sample {best_sample["global_idx"]}: '
                    f'Correlation = {best_sample["correlation"]:.4f}, '
                    f'MSE = {best_sample["mse"]:.6f}, '
                    f'MAE = {best_sample["mae"]:.6f}\n'
                    f'Red labels: Super-resolved channels | Blue labels: Available channels (should be identical)', 
                    fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for suptitle
        
        # Log to wandb if available
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"best_eeg_epoch_reconstruction": wandb.Image(fig)})
            print("Logged best EEG epoch plot to wandb")
        
        plt.savefig('best_eeg_epoch_reconstruction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create focused missing channels plot
        self._create_missing_channels_focus_plot(trainer, best_sample)
    
    def _create_missing_channels_focus_plot(self, trainer, best_sample):
        """Create a focused plot showing only the super-resolved (missing) channels"""
        lo_res = best_sample['lo_res']
        hi_res = best_sample['hi_res']
        super_res = best_sample['super_res']
        lo_res_channels = best_sample['lo_res_channels']
        hi_res_channels = best_sample['hi_res_channels']
        
        # Get missing channels
        missing_indices, missing_channels = self._get_missing_channel_indices(lo_res_channels, hi_res_channels)
        
        if not missing_indices:
            print("No missing channels to plot")
            return
        
        # Limit to reasonable number for visualization
        n_channels = min(len(missing_indices), 8)
        missing_indices = missing_indices[:n_channels]
        missing_channels = missing_channels[:n_channels]
        
        # Create figure
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2*n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]
        
        # Time axis
        time_hi = np.linspace(0, 1, hi_res.shape[1])
        
        for i, (ch_idx, ch_name) in enumerate(zip(missing_indices, missing_channels)):
            ax = axes[i]
            
            # Plot only ground truth and super-resolution for missing channels
            ax.plot(time_hi, hi_res[ch_idx], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
            ax.plot(time_hi, super_res[ch_idx], 'r-', linewidth=1.5, label='Super Resolution', alpha=0.9)
            
            # Calculate channel-specific metrics
            ch_mse = np.mean((super_res[ch_idx] - hi_res[ch_idx]) ** 2)
            ch_corr = np.corrcoef(super_res[ch_idx], hi_res[ch_idx])[0, 1] if np.std(super_res[ch_idx]) > 1e-8 and np.std(hi_res[ch_idx]) > 1e-8 else 0.0
            
            # Customize subplot
            ax.set_ylabel(f'{ch_name}\nAmplitude', fontsize=11, fontweight='bold')
            ax.set_title(f'MSE: {ch_mse:.6f}, Corr: {ch_corr:.4f}', fontsize=10, pad=5)
            ax.grid(True, alpha=0.3)
            
            # Add legend only to the first subplot
            if i == 0:
                ax.legend(loc='upper right', fontsize=10)
            
            # Set y-axis limits
            all_values = np.concatenate([hi_res[ch_idx], super_res[ch_idx]])
            y_margin = np.std(all_values) * 0.15 if np.std(all_values) > 0 else 0.1
            ax.set_ylim(np.min(all_values) - y_margin, np.max(all_values) + y_margin)
        
        # Set common x-axis label
        axes[-1].set_xlabel('Normalized Time', fontsize=12, fontweight='bold')
        
        # Add overall title
        fig.suptitle(f'Super-Resolved Channels Only - Sample {best_sample["global_idx"]}\n'
                    f'Missing Channels: {", ".join(missing_channels[:8])}{"..." if len(missing_channels) > 8 else ""}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Log to wandb if available
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"missing_channels_focus": wandb.Image(fig)})
        
        plt.savefig('missing_channels_focus.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_channel_comparison_plot(self, trainer, best_sample):
        """Create a comparison plot showing different channels in separate colors"""
        lo_res = best_sample['lo_res']
        hi_res = best_sample['hi_res']
        super_res = best_sample['super_res']
        lo_res_channels = best_sample['lo_res_channels']
        hi_res_channels = best_sample['hi_res_channels']
        
        # Get missing and common channels
        missing_indices, missing_channels = self._get_missing_channel_indices(lo_res_channels, hi_res_channels)
        common_hi_indices, common_lo_indices, common_channels = self._get_available_channel_indices(lo_res_channels, hi_res_channels)
        
        # Limit channels for clarity
        n_missing = min(len(missing_indices), 4)
        n_common = min(len(common_hi_indices), 4)
        
        missing_indices = missing_indices[:n_missing]
        missing_channels = missing_channels[:n_missing]
        common_hi_indices = common_hi_indices[:n_common]
        common_lo_indices = common_lo_indices[:n_common]
        common_channels = common_channels[:n_common]
        
        # Create figure with subplots for missing and available channels
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Time axes
        time_lo = np.linspace(0, 1, lo_res.shape[1])
        time_hi = np.linspace(0, 1, hi_res.shape[1])
        
        # Color maps
        colors_missing = plt.cm.Reds(np.linspace(0.4, 0.9, n_missing))
        colors_common = plt.cm.Blues(np.linspace(0.4, 0.9, n_common))
        
        # Plot 1: Missing channels - Ground Truth
        ax = axes[0, 0]
        for i, (ch_idx, ch_name) in enumerate(zip(missing_indices, missing_channels)):
            ax.plot(time_hi, hi_res[ch_idx], '-', color=colors_missing[i], 
                   label=ch_name, linewidth=1.5, alpha=0.8)
        ax.set_title('Missing Channels - Ground Truth', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Missing channels - Super Resolution
        ax = axes[0, 1]
        for i, (ch_idx, ch_name) in enumerate(zip(missing_indices, missing_channels)):
            ax.plot(time_hi, super_res[ch_idx], '-', color=colors_missing[i], 
                   label=ch_name, linewidth=1.5, alpha=0.8)
        ax.set_title('Missing Channels - Super Resolution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Available channels - Low Resolution Input
        ax = axes[1, 0]
        for i, (lo_idx, ch_name) in enumerate(zip(common_lo_indices, common_channels)):
            ax.plot(time_lo, lo_res[lo_idx], 'o-', color=colors_common[i], 
                   label=ch_name, markersize=3, linewidth=1.5, alpha=0.8)
        ax.set_title('Available Channels - Low Resolution Input', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Available channels - High Resolution (for comparison)
        ax = axes[1, 1]
        for i, (hi_idx, ch_name) in enumerate(zip(common_hi_indices, common_channels)):
            ax.plot(time_hi, hi_res[hi_idx], '-', color=colors_common[i], 
                   label=f'{ch_name} (GT)', linewidth=1.5, alpha=0.8)
            ax.plot(time_hi, super_res[hi_idx], '--', color=colors_common[i], 
                   label=f'{ch_name} (SR)', linewidth=1.2, alpha=0.7)
        ax.set_title('Available Channels - Ground Truth vs Super Resolution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Channel Analysis - Sample {best_sample["global_idx"]}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Log to wandb if available
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"eeg_channel_analysis": wandb.Image(fig)})
        
        plt.savefig('eeg_channel_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def on_test_end(self, trainer, pl_module):
        """Create test plots after all test processing is complete"""
        # Now try to get metrics from trainer.callback_metrics (should be available after logging)
        test_metric_names = ['test_loss', 'test_snr', 'test_mae', 'test_mse', 'test_nmse', 'test_pearson']
        
        callback_metrics = {}
        for metric_name in test_metric_names:
            value = trainer.callback_metrics.get(metric_name, 0)
            if torch.is_tensor(value):
                value = value.cpu().item()
            callback_metrics[metric_name] = value
        
        # Use callback metrics if they're non-zero, otherwise use what we collected
        if any(v != 0 for v in callback_metrics.values()):
            self.test_metrics = callback_metrics
            print(f"Using callback metrics: {self.test_metrics}")
        else:
            print(f"Using collected metrics: {self.test_metrics}")
        
        # Create test metrics plots
        self._create_test_metrics_plot(trainer)
        
        # Create best EEG epoch visualization
        self._create_best_eeg_epoch_plot(trainer)
    
    def on_train_end(self, trainer, pl_module):
        """Create training curve plots at the end of training"""
        if len(self.train_losses) > 0:
            self._create_training_curves_plot(trainer)


class SaveSuperResEpochsCallbackMinimal(pl.Callback):
    def __init__(self, save_dir, filename):
        """
        Minimal callback to save only super-resolution predictions.
        
        Args:
            save_dir (str): Directory where to save the predictions
            filename (str): Filename for the saved predictions (without .npy extension)
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.predictions = []
        
        # Create directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_test_start(self, trainer, pl_module):
        """Reset prediction list at the start of testing"""
        self.predictions = []
        print(f"Collecting super-resolution predictions to save as {self.filename}.npy...")
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect predictions from each test batch"""
        lo_res, hi_res = batch
        lo_res = (lo_res[0] if len(lo_res) == 2 else lo_res).float()
        
        with torch.no_grad():
            super_res = pl_module.super_resolution_model(lo_res)
            self.predictions.append(super_res.cpu().numpy())
    
    def on_test_end(self, trainer, pl_module):
        """Save all collected predictions"""
        if self.predictions:
            all_predictions = np.concatenate(self.predictions, axis=0)
            save_path = self.save_dir / f"{self.filename}.npy"
            np.save(save_path, all_predictions)
            print(f"Saved {all_predictions.shape[0]} super-resolution epochs to: {save_path}")
            self.predictions.clear()
        else:
            print("Warning: No predictions to save!")