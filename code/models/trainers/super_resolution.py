import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torchmetrics.aggregation import MeanMetric
from torchmetrics.audio import SignalNoiseRatio
from models.definers.super_resolution import EEGSuperResolutionDefiner as EEGSuperResolution
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, NormalizedRootMeanSquaredError, PearsonCorrCoef

class SuperResolutionTrainerModel(pl.LightningModule):
    def __init__(self, lo_res_channel_names, hi_res_channel_names, time_steps, lr=5e-5, weight_decay=0.5, beta1=0.9, beta2=0.95):
        super().__init__()
        self.super_resolution_model = EEGSuperResolution(lo_res_channel_names, hi_res_channel_names, time_steps)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
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
        self.mean_test_nrmse = NormalizedRootMeanSquaredError()
        self.mean_test_pearson = PearsonCorrCoef()

    def training_step(self, batch, batch_idx):
        lo_res, hi_res = batch
        lo_res = (lo_res[0] if len(lo_res) == 2 else lo_res).float()
        hi_res = (hi_res[0] if len(hi_res) == 2 else hi_res).float()
        super_res = self.super_resolution_model(lo_res)
        loss = self.super_resolution_model.compute_loss(hi_res, super_res)
        
        # Update metrics
        self.mean_train_loss(loss)
        self.mean_train_mae(super_res, hi_res)
        
        self.log("train_batch_loss", self.mean_train_loss)
        return loss

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
        self.mean_test_nrmse(super_res, hi_res)
        self.mean_test_pearson(super_res, hi_res)
        
        return loss

    def on_test_epoch_end(self):
        # Log all test metrics
        test_loss = self.mean_test_loss.compute()
        test_snr = self.mean_test_snr.compute()
        test_mae = self.mean_test_mae.compute()
        test_mse = self.mean_test_mse.compute()
        test_nrmse = self.mean_test_nrmse.compute()
        test_pearson = self.mean_test_pearson.compute()
        
        self.log("test_loss", test_loss)
        self.log("test_snr", test_snr)
        self.log("test_mae", test_mae)
        self.log("test_mse", test_mse)
        self.log("test_nrmse", test_nrmse)
        self.log("test_pearson", test_pearson)
        
        # Print test results
        print(f"\n=== SUPER RESOLUTION TEST RESULTS ===")
        print(f"Test Loss: {test_loss:.6f}")
        print(f"Test SNR: {test_snr:.4f} dB")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Test MSE: {test_mse:.6f}")
        print(f"Test NRMSE: {test_nrmse:.6f}")
        print(f"Test Pearson Correlation: {test_pearson:.4f}")
        print(f"=====================================\n")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))
        return optimizer


class SuperResolutionPlottingCallback(pl.Callback):
    def __init__(self):
        # Training curves data
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        
        # Test metrics data
        self.test_metrics = {}
    
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
    
    def on_test_epoch_end(self, trainer, pl_module):
        # Collect all test metrics
        test_metric_names = ['test_loss', 'test_snr', 'test_mae', 'test_mse', 'test_nrmse', 'test_pearson']
        
        for metric_name in test_metric_names:
            value = trainer.callback_metrics.get(metric_name, 0)
            if torch.is_tensor(value):
                value = value.cpu().item()
            self.test_metrics[metric_name] = value
        
        # Create test metrics bar plot
        self._create_test_metrics_plot(trainer)
    
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
        
        # plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
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
        
        # plt.savefig('training_mae_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_test_metrics_plot(self, trainer):
        """Create bar plot for test metrics"""
        if not self.test_metrics:
            return
        
        # Prepare data for plotting
        metric_names = list(self.test_metrics.keys())
        metric_values = list(self.test_metrics.values())
        
        # Create more readable labels
        display_names = {
            'test_loss': 'Loss',
            'test_mae': 'MAE',
            'test_mse': 'MSE', 
            'test_nrmse': 'NRMSE',
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
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            elif metric_names[i] == 'test_snr':
                # Format SNR values
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            else:
                # Format other metrics in scientific notation if very small
                if abs(value) < 0.001:
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.2e}', ha='center', va='bottom', fontweight='bold')
                else:
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Log to wandb if available
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"test_metrics_bar_plot": wandb.Image(fig)})
        
        # plt.savefig('test_metrics_bar_plot.png', dpi=300, bbox_inches='tight')
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
            'test_nrmse': ('Normalized RMSE', f"{self.test_metrics['test_nrmse']:.6f}", 'Lower is better'),
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
        
        # plt.savefig('test_metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def on_train_end(self, trainer, pl_module):
        """Create training curve plots at the end of training"""
        if len(self.train_losses) > 0:
            self._create_training_curves_plot(trainer)