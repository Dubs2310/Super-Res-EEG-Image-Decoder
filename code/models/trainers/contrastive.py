import wandb 
import torch
import random
import numpy as np
from torch.optim import Adam
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchmetrics import MeanMetric
from torchmetrics.classification import MultilabelConfusionMatrix
from utils.singletons.coco import COCO
from models.definers.encoder import EEGEncoderDefiner as EEGEncoder
from torchmetrics.classification import (
    MultilabelAccuracy, 
    MultilabelF1Score, 
    MultilabelPrecision, 
    MultilabelRecall, 
    MultilabelAUROC
)

class ContrastiveTrainerModel(pl.LightningModule):
    def __init__(self, num_channels, timesteps, num_fine_labels=80, alpha=0.99, learning_rate=1e-3):
        super().__init__()
        self.encoder = EEGEncoder(num_channels, timesteps)

        coco = COCO()
        _, train_img_features, _, train_img_id_to_indices = coco.get_train_set()
        _, test_img_features, _, test_img_id_to_indices = coco.get_test_set()

        # Store features and ensure they can be moved to the correct device
        self.register_buffer('train_image_features', train_img_features.cpu())
        self.register_buffer('test_image_features', test_img_features.cpu())
        
        # Store the mappings
        self.train_img_id_to_indices = train_img_id_to_indices
        self.test_img_id_to_indices = test_img_id_to_indices
        
        self.num_fine_labels = num_fine_labels
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['train_features', 'test_features'])
        
        # Create class-wise feature representations
        self._create_class_features()
        
        # Training metrics
        self.train_loss = MeanMetric()
        self.train_accuracy = MultilabelAccuracy(num_labels=num_fine_labels)
        
        # Validation metrics
        self.val_loss = MeanMetric()
        self.val_accuracy = MultilabelAccuracy(num_labels=num_fine_labels)
        
        # K-way accuracy metrics
        self.v2_acc = MultilabelAccuracy(num_labels=2)
        self.v4_acc = MultilabelAccuracy(num_labels=4)
        self.v10_acc = MultilabelAccuracy(num_labels=10)
        self.v80_acc = MultilabelAccuracy(num_labels=80)
        self.top5_acc = MeanMetric()
        
        # Additional test metrics
        self.test_precision = MultilabelPrecision(num_labels=num_fine_labels)
        self.test_recall = MultilabelRecall(num_labels=num_fine_labels)
        self.test_f1 = MultilabelF1Score(num_labels=num_fine_labels)
        self.test_auroc = MultilabelAUROC(num_labels=num_fine_labels)
        
        # Confusion matrix for test set
        self.test_confusion_matrix = MultilabelConfusionMatrix(num_labels=num_fine_labels)
        
        # Best model tracking
        self.best_accuracy = 0.0
        self.best_model_weights = None
        self.best_epoch_info = {}

    def _create_class_features(self):
        """Create representative features for each class by averaging all features for that class"""
        # For now, let's use the first 80 features as class representatives
        # You might need to modify this based on your actual feature organization
        feature_dim = self.train_image_features.shape[1]
        
        # Initialize class features
        train_class_features = torch.zeros(self.num_fine_labels, feature_dim)
        test_class_features = torch.zeros(self.num_fine_labels, feature_dim)
        
        # Simple approach: use first 80 features as class representatives
        for i in range(self.num_fine_labels):
            if i < self.train_image_features.shape[0]:
                train_class_features[i] = self.train_image_features[i]
            if i < self.test_image_features.shape[0]:
                test_class_features[i] = self.test_image_features[i]
        
        self.register_buffer('train_class_features', train_class_features)
        self.register_buffer('test_class_features', test_class_features)

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        eeg_data, output = batch
        img, img_features, text_features, super_labels, fine_labels = output
        
        # Ensure all tensors are on the same device
        img_features = img_features.to(self.device)
        text_features = text_features.to(self.device)
        eeg_features = self.encoder(eeg_data).float()
        img_loss = self.encoder.compute_loss(eeg_features, img_features)
        text_loss = self.encoder.compute_loss(eeg_features, text_features)
        loss = self.alpha * img_loss + (1 - self.alpha) * text_loss
        
        # Use pre-computed class features for training evaluation
        logits_img = self.encoder.logit_scale * eeg_features @ self.train_class_features.to(self.device).T
        logits_single = logits_img

        preds = torch.sigmoid(logits_single)
        targets = fine_labels.float()        
        self.train_loss.update(loss)
        self.train_accuracy.update(preds, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        eeg_data, output = batch
        img, img_features, text_features, super_labels, fine_labels = output
        
        # Ensure all tensors are on the same device
        img_features = img_features.to(self.device)
        text_features = text_features.to(self.device)
        eeg_features = self.encoder(eeg_data).float()
        img_loss = self.encoder.compute_loss(eeg_features, img_features)
        text_loss = self.encoder.compute_loss(eeg_features, text_features)
        loss = img_loss * self.alpha + text_loss * (1 - self.alpha)
        self.val_loss.update(loss)
        
        all_fine_labels = set(range(self.num_fine_labels))
        
        for k, acc_metric in [(2, self.v2_acc), (4, self.v4_acc), (10, self.v10_acc), (80, self.v80_acc)]:
            for idx in range(eeg_features.shape[0]):
                active_fine_labels = torch.where(fine_labels[idx] == 1)[0].tolist()
                
                if len(active_fine_labels) > 0:
                    remaining_labels = list(all_fine_labels - set(active_fine_labels))
                    num_negatives = min(k - len(active_fine_labels), len(remaining_labels))
                    
                    if num_negatives > 0:
                        selected_negatives = random.sample(remaining_labels, num_negatives)
                        selected_classes = active_fine_labels + selected_negatives
                    else:
                        selected_classes = active_fine_labels[:k]
                    
                    # Move test features to device and ensure correct dtype
                    selected_img_features = self.test_image_features[selected_classes].to(self.device).float()
                    logits_img = self.encoder.logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    preds = torch.sigmoid(logits_single).unsqueeze(0)
                    targets = torch.zeros(1, len(selected_classes), device=self.device)
                    
                    for j, class_idx in enumerate(selected_classes):
                        if class_idx in active_fine_labels:
                            targets[0, j] = 1
                    
                    if k == 80:
                        _, top5_indices = torch.topk(logits_single, min(5, len(logits_single)), largest=True)
                        top5_classes = [selected_classes[i] for i in top5_indices.tolist()]
                        has_true_in_top5 = any(class_idx in active_fine_labels for class_idx in top5_classes)
                        self.top5_acc.update(float(has_true_in_top5))
                    
                    acc_metric.update(preds, targets)
        
        # Use pre-computed class features for validation
        logits_img = self.encoder.logit_scale * eeg_features @ self.test_class_features.to(self.device).T
        preds = torch.sigmoid(logits_img)
        targets = fine_labels.float()
        self.val_accuracy.update(preds, targets)
        return loss

    def test_step(self, batch, batch_idx):
        eeg_data, output = batch
        img, img_features, text_features, super_labels, fine_labels = output
        
        # Ensure all tensors are on the same device
        img_features = img_features.to(self.device)
        text_features = text_features.to(self.device)
        eeg_features = self.encoder(eeg_data).float()
        img_loss = self.encoder.compute_loss(eeg_features, img_features)
        text_loss = self.encoder.compute_loss(eeg_features, text_features)
        loss = img_loss * self.alpha + text_loss * (1 - self.alpha)
        
        # Use pre-computed class features for test evaluation
        logits_img = self.encoder.logit_scale * eeg_features @ self.test_class_features.to(self.device).T
        preds = torch.sigmoid(logits_img)
        targets = fine_labels.float()
        
        # Update all test metrics
        self.test_precision.update(preds, targets)
        self.test_recall.update(preds, targets)
        self.test_f1.update(preds, targets)
        self.test_auroc.update(preds, targets)
        self.test_confusion_matrix.update(preds, targets.int())
        
        # Also update validation metrics for consistency
        self.val_loss.update(loss)
        self.val_accuracy.update(preds, targets)
        
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        self.log('train_loss', self.train_loss.compute(), prog_bar=True)
        self.log('train_accuracy', self.train_accuracy.compute(), prog_bar=True)
        self.train_loss.reset()
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        val_accuracy = self.val_accuracy.compute()
        v2_accuracy = self.v2_acc.compute()
        v4_accuracy = self.v4_acc.compute()
        v10_accuracy = self.v10_acc.compute()
        v80_accuracy = self.v80_acc.compute()
        top5_accuracy = self.top5_acc.compute()
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_accuracy', val_accuracy, prog_bar=True)
        self.log('v2_acc', v2_accuracy)
        self.log('v4_acc', v4_accuracy)
        self.log('v10_acc', v10_accuracy)
        self.log('v80_acc', v80_accuracy)
        self.log('top5_acc', top5_accuracy)
        
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_model_weights = self.state_dict().copy()
            self.best_epoch_info = {
                "epoch": self.current_epoch + 1,
                "train_loss": self.train_loss.compute(),
                "train_accuracy": self.train_accuracy.compute(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "v2_acc": v2_accuracy,
                "v4_acc": v4_accuracy,
                "v10_acc": v10_accuracy,
                "v80_acc": v80_accuracy
            }
        
        self.val_loss.reset()
        self.val_accuracy.reset()
        self.v2_acc.reset()
        self.v4_acc.reset()
        self.v10_acc.reset()
        self.v80_acc.reset()
        self.top5_acc.reset()

    def on_test_epoch_end(self):
        """Log additional test metrics at the end of testing"""
        test_precision = self.test_precision.compute()
        test_recall = self.test_recall.compute()
        test_f1 = self.test_f1.compute()
        test_auroc = self.test_auroc.compute()
        
        self.log('test_precision', test_precision)
        self.log('test_recall', test_recall)
        self.log('test_f1', test_f1)
        self.log('test_auroc', test_auroc)
        
        # Print detailed test results
        print(f"\n=== TEST RESULTS ===")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        print(f"Test AUROC: {test_auroc:.4f}")
        print(f"====================\n")
        
        # Reset test metrics
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.test_confusion_matrix.reset()


class PlottingCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.v2_accs = []
        self.v4_accs = []
        self.v10_accs = []
        self.v80_accs = []
        
        # For t-SNE visualization
        self.test_embeddings = []
        self.test_labels = []
        self.test_super_labels = []
        
        # For confusion matrix visualization
        self.test_predictions = []
        self.test_true_labels = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss', 0)
        train_accuracy = trainer.callback_metrics.get('train_accuracy', 0)
        
        # Convert to CPU and then to Python float if they are tensors
        if torch.is_tensor(train_loss):
            train_loss = train_loss.cpu().item()
        if torch.is_tensor(train_accuracy):
            train_accuracy = train_accuracy.cpu().item()
            
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect embeddings, predictions, and labels during test phase"""
        eeg_data, output = batch
        img, img_features, text_features, super_labels, fine_labels = output
        
        # Get embeddings before sigmoid (raw model output)
        with torch.no_grad():
            embeddings = pl_module.encoder(eeg_data)
            
            # Get predictions using the same logic as in test_step
            logits_img = pl_module.encoder.logit_scale * embeddings @ pl_module.test_class_features.to(pl_module.device).T
            predictions = torch.sigmoid(logits_img)
            
            # Convert to binary predictions (threshold at 0.5)
            binary_predictions = (predictions > 0.5).float()
        
        # Store data for visualizations
        self.test_embeddings.append(embeddings.cpu().numpy())
        self.test_labels.append(fine_labels.cpu().numpy())
        self.test_super_labels.append(super_labels.cpu().numpy())
        self.test_predictions.append(binary_predictions.cpu().numpy())
        self.test_true_labels.append(fine_labels.cpu().numpy())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss', 0)
        val_accuracy = trainer.callback_metrics.get('val_accuracy', 0)
        v2_acc = trainer.callback_metrics.get('v2_acc', 0)
        v4_acc = trainer.callback_metrics.get('v4_acc', 0)
        v10_acc = trainer.callback_metrics.get('v10_acc', 0)
        v80_acc = trainer.callback_metrics.get('v80_acc', 0)
        
        # Convert all tensors to CPU and then to Python floats
        metrics = [val_loss, val_accuracy, v2_acc, v4_acc, v10_acc, v80_acc]
        converted_metrics = []
        for metric in metrics:
            if torch.is_tensor(metric):
                converted_metrics.append(metric.cpu().item())
            else:
                converted_metrics.append(metric)
        
        val_loss, val_accuracy, v2_acc, v4_acc, v10_acc, v80_acc = converted_metrics
        
        self.test_losses.append(val_loss)
        self.test_accuracies.append(val_accuracy)
        self.v2_accs.append(v2_acc)
        self.v4_accs.append(v4_acc)
        self.v10_accs.append(v10_acc)
        self.v80_accs.append(v80_acc)
    
    def _create_and_log_plot(self, data1, data2, labels, title, filename, trainer):
        """Helper function to create individual plots"""
        plt.figure(figsize=(10, 6))
        if data2 is not None:
            plt.plot(data1, label=labels[0])
            plt.plot(data2, label=labels[1])
        else:
            plt.plot(data1, label=labels[0])
        plt.legend()
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(title.split()[-1])  # Use the last word as y-label
        plt.grid(True, alpha=0.3)
        
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({filename: wandb.Image(plt)})
        
        # plt.savefig(f'{filename}.png')
        plt.close()
    
    def _create_best_model_info_plot(self, pl_module, trainer):
        """Create a plot showing the best model information"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        info_text = (f"Best Model Info (from Epoch {pl_module.best_epoch_info.get('epoch', 0)}):\n\n"
                    f"Train Loss: {pl_module.best_epoch_info.get('train_loss', 0):.4f}\n"
                    f"Train Accuracy: {pl_module.best_epoch_info.get('train_accuracy', 0):.4f}\n\n"
                    f"Val Loss: {pl_module.best_epoch_info.get('val_loss', 0):.4f}\n"
                    f"Val Accuracy: {pl_module.best_epoch_info.get('val_accuracy', 0):.4f}\n\n"
                    f"2-way Accuracy: {pl_module.best_epoch_info.get('v2_acc', 0):.4f}\n"
                    f"4-way Accuracy: {pl_module.best_epoch_info.get('v4_acc', 0):.4f}\n"
                    f"10-way Accuracy: {pl_module.best_epoch_info.get('v10_acc', 0):.4f}\n"
                    f"80-way Accuracy: {pl_module.best_epoch_info.get('v80_acc', 0):.4f}")
        
        ax.text(0.5, 0.5, info_text, fontsize=14, ha='center', va='center', 
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.title('Best Model Performance Summary', fontsize=16, pad=20)
        
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"best_model_info": wandb.Image(fig)})
        
        # plt.savefig('best_model_info.png')
        plt.close()
    
    def _create_tsne_plot(self, pl_module, trainer):
        """Create t-SNE visualization of test embeddings"""
        if not self.test_embeddings:
            print("No test embeddings available for t-SNE visualization")
            return
        
        # Concatenate all embeddings and labels
        all_embeddings = np.concatenate(self.test_embeddings, axis=0)
        all_fine_labels = np.concatenate(self.test_labels, axis=0)
        all_super_labels = np.concatenate(self.test_super_labels, axis=0)
        
        print(f"Creating t-SNE visualization with {all_embeddings.shape[0]} samples...")
        
        # Limit number of samples for t-SNE if too many (for computational efficiency)
        max_samples = 2000
        if all_embeddings.shape[0] > max_samples:
            indices = np.random.choice(all_embeddings.shape[0], max_samples, replace=False)
            all_embeddings = all_embeddings[indices]
            all_fine_labels = all_fine_labels[indices]
            all_super_labels = all_super_labels[indices]
            print(f"Subsampled to {max_samples} samples for t-SNE")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(all_embeddings)
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, all_embeddings.shape[0] - 1))
        embeddings_2d = tsne.fit_transform(embeddings_scaled)
        
        # Get COCO categories for labeling
        coco = COCO()
        fine_categories = coco.get_fine_categories()
        super_categories = coco.get_super_categories()
        
        # Create t-SNE plot colored by super categories
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Colored by super categories
        unique_super_labels = np.unique(all_super_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_super_labels)))
        
        for i, super_label in enumerate(unique_super_labels):
            mask = all_super_labels == super_label
            if np.any(mask):
                super_name = super_categories[super_label] if super_label < len(super_categories) else f'Super_{super_label}'
                ax1.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=super_name, alpha=0.7, s=20)
        
        ax1.set_title('t-SNE of EEG Embeddings (Colored by Super Categories)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Colored by most frequent fine category per sample
        # For multilabel data, use the first active label for coloring
        sample_fine_labels = []
        for fine_label_vector in all_fine_labels:
            active_labels = np.where(fine_label_vector == 1)[0]
            if len(active_labels) > 0:
                sample_fine_labels.append(active_labels[0])  # Use first active label
            else:
                sample_fine_labels.append(-1)  # No active label
        
        sample_fine_labels = np.array(sample_fine_labels)
        unique_fine_labels = np.unique(sample_fine_labels)
        unique_fine_labels = unique_fine_labels[unique_fine_labels >= 0]  # Remove -1 (no label)
        
        # Limit to top 20 most frequent fine categories for visibility
        if len(unique_fine_labels) > 20:
            label_counts = [(label, np.sum(sample_fine_labels == label)) for label in unique_fine_labels]
            label_counts.sort(key=lambda x: x[1], reverse=True)
            unique_fine_labels = np.array([label for label, _ in label_counts[:20]])
        
        colors_fine = plt.cm.tab20(np.linspace(0, 1, len(unique_fine_labels)))
        
        for i, fine_label in enumerate(unique_fine_labels):
            mask = sample_fine_labels == fine_label
            if np.any(mask):
                fine_name = fine_categories[fine_label] if fine_label < len(fine_categories) else f'Fine_{fine_label}'
                ax2.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors_fine[i]], label=fine_name, alpha=0.7, s=20)
        
        # Plot samples with no labels in gray
        no_label_mask = sample_fine_labels == -1
        if np.any(no_label_mask):
            ax2.scatter(embeddings_2d[no_label_mask, 0], embeddings_2d[no_label_mask, 1], 
                       c='gray', label='No Label', alpha=0.5, s=20)
        
        ax2.set_title('t-SNE of EEG Embeddings (Colored by Fine Categories - Top 20)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"tsne_embeddings": wandb.Image(fig)})
        
        # plt.savefig('tsne_embeddings.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a simpler density plot
        self._create_tsne_density_plot(embeddings_2d, trainer)
        
        print("t-SNE visualization completed!")
    
    def _create_tsne_density_plot(self, embeddings_2d, trainer):
        """Create a density plot of the t-SNE embeddings"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a hexbin plot for density visualization
        hb = ax.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=30, cmap='Blues', alpha=0.8)
        
        ax.set_title('t-SNE Embedding Density Plot', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        
        # Add colorbar
        plt.colorbar(hb, ax=ax, label='Point Density')
        
        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"tsne_density": wandb.Image(fig)})
        
        # plt.savefig('tsne_density.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confusion_matrix_plot(self, pl_module, trainer):
        """Create per-class confusion matrix plot for multilabel classification"""
        if not self.test_predictions or not self.test_true_labels:
            print("No test predictions available for confusion matrix visualization")
            return
        
        # Concatenate all predictions and labels
        y_pred = np.concatenate(self.test_predictions, axis=0)
        y_true = np.concatenate(self.test_true_labels, axis=0)
        
        num_classes = y_pred.shape[1]
        
        # Get COCO fine categories for labels
        coco = COCO()
        fine_categories = coco.get_fine_categories()
        
        # Calculate grid dimensions
        if num_classes <= 16:
            # For 16 or fewer classes, use 4x4 grid
            cols = 4
            rows = int(np.ceil(num_classes / cols))
        elif num_classes <= 25:
            # For 17-25 classes, use 5x5 grid
            cols = 5
            rows = int(np.ceil(num_classes / cols))
        elif num_classes <= 64:
            # For 26-64 classes, use 8x8 grid
            cols = 8
            rows = int(np.ceil(num_classes / cols))
        else:
            # For more than 64 classes, use 10x10 grid
            cols = 10
            rows = int(np.ceil(num_classes / cols))
        
        # Create the figure
        figsize = (cols * 3, rows * 2.5)
        f, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Handle case where we have only one row or column
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.ravel()
        
        # Create confusion matrices for each class
        displays = []
        for i in range(num_classes):
            # Get class name
            class_name = fine_categories[i] if i < len(fine_categories) else f'Class {i}'
            class_name = class_name[:15] + '...' if len(class_name) > 15 else class_name  # Truncate long names
            
            # Create confusion matrix for this class
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            
            # Plot on the corresponding axis
            ax = axes[i] if len(axes) > i else None
            if ax is not None:
                disp.plot(ax=ax, values_format='.4g', cmap='Blues')
                disp.ax_.set_title(f'{class_name}', fontsize=10, pad=5)
                
                # Remove x-label for all but bottom row
                if i < num_classes - cols:
                    disp.ax_.set_xlabel('')
                
                # Remove y-label for all but leftmost column
                if i % cols != 0:
                    disp.ax_.set_ylabel('')
                
                # Remove individual colorbars
                if hasattr(disp, 'im_') and disp.im_.colorbar:
                    disp.im_.colorbar.remove()
                
                displays.append(disp)
        
        # Hide unused subplots
        for i in range(num_classes, len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        
        # Add a single colorbar for all subplots
        if displays:
            f.colorbar(displays[0].im_, ax=axes, shrink=0.8, aspect=30)
        
        # Add overall title
        plt.suptitle('Per-Class Confusion Matrices (Multilabel Classification)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"confusion_matrices_per_class": wandb.Image(f)})
        
        # plt.savefig('confusion_matrices_per_class.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics plot
        self._create_confusion_matrix_summary_stats(y_true, y_pred, fine_categories, trainer)
        
        print("Confusion matrix visualization completed!")
    
    def _create_confusion_matrix_summary_stats(self, y_true, y_pred, fine_categories, trainer):
        """Create a summary plot with precision, recall, F1 for each class"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        num_classes = y_true.shape[1]
        
        # Calculate metrics for each class
        precisions = []
        recalls = []
        f1_scores = []
        class_names = []
        
        for i in range(num_classes):
            precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            
            class_name = fine_categories[i] if i < len(fine_categories) else f'Class {i}'
            class_names.append(class_name)
        
        # Create bar plot
        x = np.arange(num_classes)
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(max(15, num_classes * 0.5), 8))
        
        bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8, color='lightcoral')
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics (Precision, Recall, F1-Score)')
        ax.set_xticks(x)
        ax.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in class_names], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only show label if bar is tall enough
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"per_class_metrics_summary": wandb.Image(fig)})
        
        # plt.savefig('per_class_metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def on_train_end(self, trainer, pl_module):
        # Create separate plots for each metric
        self._create_and_log_plot(
            self.train_losses, self.test_losses, 
            ['Train Loss', 'Val Loss'], 
            'Loss Curves', 'loss_curves', trainer
        )
        
        self._create_and_log_plot(
            self.train_accuracies, self.test_accuracies, 
            ['Train Accuracy', 'Val Accuracy'], 
            'Accuracy Curves', 'accuracy_curves', trainer
        )
        
        self._create_and_log_plot(
            self.v2_accs, None, 
            ['2-way Accuracy'], 
            '2-way Accuracy Curve', 'v2_accuracy_curve', trainer
        )
        
        self._create_and_log_plot(
            self.v4_accs, None, 
            ['4-way Accuracy'], 
            '4-way Accuracy Curve', 'v4_accuracy_curve', trainer
        )
        
        self._create_and_log_plot(
            self.v10_accs, None, 
            ['10-way Accuracy'], 
            '10-way Accuracy Curve', 'v10_accuracy_curve', trainer
        )
        
        self._create_and_log_plot(
            self.v80_accs, None, 
            ['80-way Accuracy'], 
            '80-way Accuracy Curve', 'v80_accuracy_curve', trainer
        )
        
        # Create best model info plot
        self._create_best_model_info_plot(pl_module, trainer)
        
        # Create confusion matrix plot (only if test metrics are available)
        if hasattr(pl_module, 'test_confusion_matrix') and pl_module.test_confusion_matrix.confmat is not None:
            self._create_confusion_matrix_plot(pl_module, trainer)
    
    def on_test_end(self, trainer, pl_module):
        """Create confusion matrix plot and t-SNE visualization after test phase"""
        self._create_confusion_matrix_plot(pl_module, trainer)
        self._create_tsne_plot(pl_module, trainer)