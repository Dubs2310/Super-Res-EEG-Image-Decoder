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
        targets_float = fine_labels.float()
        targets_int = fine_labels.int()
        
        # Update all test metrics with appropriate data types
        self.test_precision.update(preds, targets_int)
        self.test_recall.update(preds, targets_int)
        self.test_f1.update(preds, targets_int)
        self.test_auroc.update(preds, targets_int)
        self.test_confusion_matrix.update(preds, targets_int)
        
        # Also update validation metrics for consistency (these expect float)
        self.val_loss.update(loss)
        self.val_accuracy.update(preds, targets_float)
        
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
        
        # For confusion matrix visualization
        self.test_predictions = []
        self.test_true_labels = []
        self.test_prediction_probs = []  # Store probabilities for PR curves
    
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
        """Collect predictions and labels during test phase"""
        eeg_data, output = batch
        img, img_features, text_features, super_labels, fine_labels = output
        
        # Get predictions using the same logic as in test_step
        with torch.no_grad():
            embeddings = pl_module.encoder(eeg_data)
            logits_img = pl_module.encoder.logit_scale * embeddings @ pl_module.test_class_features.to(pl_module.device).T
            prediction_probs = torch.sigmoid(logits_img)
            
            # Convert to binary predictions (threshold at 0.5)
            binary_predictions = (prediction_probs > 0.5).float()
        
        # Store data for visualizations
        self.test_predictions.append(binary_predictions.cpu().numpy())
        self.test_true_labels.append(fine_labels.cpu().numpy())
        self.test_prediction_probs.append(prediction_probs.cpu().numpy())
    
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
        
        plt.close()
    
    def _create_confusion_matrix_plot(self, pl_module, trainer):
        """Create per-class 2x2 confusion matrix plots for multilabel classification"""
        if not self.test_predictions or not self.test_true_labels:
            print("No test predictions available for confusion matrix visualization")
            return
        
        # Concatenate all predictions and labels
        y_pred = np.concatenate(self.test_predictions, axis=0)
        y_true = np.concatenate(self.test_true_labels, axis=0)
        
        num_classes = y_pred.shape[1]
        
        # Get COCO fine categories for labels
        coco = COCO()
        fine_categories = coco.all_fine_categories
        
        # Calculate grid dimensions for 2x2 confusion matrices
        if num_classes <= 16:
            cols = 4
            rows = int(np.ceil(num_classes / cols))
        elif num_classes <= 25:
            cols = 5
            rows = int(np.ceil(num_classes / cols))
        elif num_classes <= 64:
            cols = 8
            rows = int(np.ceil(num_classes / cols))
        else:
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
        
        # Create 2x2 confusion matrices for each class
        for i in range(num_classes):
            # Get class name
            class_name = fine_categories[i] if i < len(fine_categories) else f'Class {i}'
            class_name = class_name[:15] + '...' if len(class_name) > 15 else class_name
            
            # Create confusion matrix for this binary classification task
            cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
            
            # Plot on the corresponding axis
            ax = axes[i] if len(axes) > i else None
            if ax is not None:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
                disp.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
                ax.set_title(f'{class_name}', fontsize=10, pad=5)
                
                # Remove x-label for all but bottom row
                if i < num_classes - cols:
                    ax.set_xlabel('')
                
                # Remove y-label for all but leftmost column
                if i % cols != 0:
                    ax.set_ylabel('')
        
        # Hide unused subplots
        for i in range(num_classes, len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        
        # Add overall title
        plt.suptitle('Per-Class 2x2 Confusion Matrices (Multilabel Classification)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"confusion_matrices_per_class": wandb.Image(f)})
        
        plt.close()
        print("Confusion matrix visualization completed!")
    
    def _create_classification_report_plot(self, pl_module, trainer):
        """Create a classification report visualization"""
        if not self.test_predictions or not self.test_true_labels:
            print("No test predictions available for classification report")
            return
        
        from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
        
        # Concatenate all predictions and labels
        y_pred = np.concatenate(self.test_predictions, axis=0)
        y_true = np.concatenate(self.test_true_labels, axis=0)
        
        num_classes = y_pred.shape[1]
        
        # Get COCO fine categories for labels
        coco = COCO()
        fine_categories = coco.all_fine_categories
        
        # Calculate metrics for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Calculate accuracy for each class (per-label accuracy)
        accuracy = []
        for i in range(num_classes):
            acc = accuracy_score(y_true[:, i], y_pred[:, i])
            accuracy.append(acc)
        accuracy = np.array(accuracy)
        
        # Calculate macro and micro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        macro_accuracy = np.mean(accuracy)
        
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        # Micro accuracy for multilabel is the same as Hamming loss complement
        micro_accuracy = accuracy_score(y_true, y_pred)
        
        # Create the classification report table
        class_names = [fine_categories[i] if i < len(fine_categories) else f'Class {i}' 
                      for i in range(num_classes)]
        
        # Sort classes by precision (highest first)
        sorted_indices = np.argsort(precision)[::-1]
        
        # Create sorted arrays
        sorted_precision = precision[sorted_indices]
        sorted_recall = recall[sorted_indices]
        sorted_f1 = f1[sorted_indices]
        sorted_accuracy = accuracy[sorted_indices]
        sorted_support = support[sorted_indices]
        sorted_class_names = [class_names[i] for i in sorted_indices]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, max(12, num_classes * 0.3)))
        
        # Plot 1: Bar chart of metrics per class (sorted by precision)
        x = np.arange(num_classes)
        width = 0.2
        
        bars1 = ax1.bar(x - 1.5*width, sorted_precision, width, label='Precision', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x - 0.5*width, sorted_recall, width, label='Recall', alpha=0.8, color='lightcoral')
        bars3 = ax1.bar(x + 0.5*width, sorted_f1, width, label='F1-Score', alpha=0.8, color='lightgreen')
        bars4 = ax1.bar(x + 1.5*width, sorted_accuracy, width, label='Accuracy', alpha=0.8, color='gold')
        
        ax1.set_xlabel('Classes (Sorted by Precision - Highest to Lowest)')
        ax1.set_ylabel('Score')
        ax1.set_title('Per-Class Performance Metrics (Sorted by Precision)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in sorted_class_names], 
                           rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.05)
        
        # Add macro average lines
        ax1.axhline(y=macro_precision, color='blue', linestyle='--', alpha=0.7, label=f'Macro Avg Precision: {macro_precision:.3f}')
        ax1.axhline(y=macro_recall, color='red', linestyle='--', alpha=0.7, label=f'Macro Avg Recall: {macro_recall:.3f}')
        ax1.axhline(y=macro_f1, color='green', linestyle='--', alpha=0.7, label=f'Macro Avg F1: {macro_f1:.3f}')
        ax1.axhline(y=macro_accuracy, color='orange', linestyle='--', alpha=0.7, label=f'Macro Avg Accuracy: {macro_accuracy:.3f}')
        ax1.legend(loc='upper right')
        
        # Plot 2: Table with detailed metrics (top 20 classes by precision)
        ax2.axis('tight')
        ax2.axis('off')
        
        # Create table data for display (showing top 20 classes by precision + averages)
        display_rows = min(20, num_classes)
        table_data = []
        
        for i in range(display_rows):
            table_data.append([
                sorted_class_names[i][:15] + '...' if len(sorted_class_names[i]) > 15 else sorted_class_names[i],
                f'{sorted_precision[i]:.3f}',
                f'{sorted_recall[i]:.3f}',
                f'{sorted_f1[i]:.3f}',
                f'{sorted_accuracy[i]:.3f}',
                f'{int(sorted_support[i])}'
            ])
        
        # Add average rows
        table_data.append(['', '', '', '', '', ''])  # Empty row
        table_data.append([
            'macro avg',
            f'{macro_precision:.3f}',
            f'{macro_recall:.3f}',
            f'{macro_f1:.3f}',
            f'{macro_accuracy:.3f}',
            f'{int(np.sum(support))}'
        ])
        table_data.append([
            'micro avg',
            f'{micro_precision:.3f}',
            f'{micro_recall:.3f}',
            f'{micro_f1:.3f}',
            f'{micro_accuracy:.3f}',
            f'{int(np.sum(support))}'
        ])
        
        table = ax2.table(cellText=table_data,
                         colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'Support'],
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        # Color the header
        for i in range(6):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color the average rows
        for i in range(6):
            table[(len(table_data) - 2, i)].set_facecolor('#f0f0f0')  # macro avg
            table[(len(table_data) - 1, i)].set_facecolor('#e0e0e0')  # micro avg
        
        ax2.set_title(f'Classification Report (Top {display_rows} classes by Precision)', 
                     fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"classification_report": wandb.Image(fig)})
        
        plt.close()
        
        # Print detailed classification report to console
        print("\n" + "="*90)
        print("DETAILED CLASSIFICATION REPORT (Top 20 Classes by Precision)")
        print("="*90)
        for i in range(min(20, num_classes)):
            orig_idx = sorted_indices[i]
            class_name = class_names[orig_idx]
            print(f"Class {orig_idx:2d} - {class_name:25s}: P={sorted_precision[i]:.3f}, R={sorted_recall[i]:.3f}, F1={sorted_f1[i]:.3f}, Acc={sorted_accuracy[i]:.3f}, Support={int(sorted_support[i])}")
        
        print("\nAVERAGES:")
        print(f"Macro Average: P={macro_precision:.3f}, R={macro_recall:.3f}, F1={macro_f1:.3f}, Acc={macro_accuracy:.3f}")
        print(f"Micro Average: P={micro_precision:.3f}, R={micro_recall:.3f}, F1={micro_f1:.3f}, Acc={micro_accuracy:.3f}")
        print("="*90)
        
        print("Classification report visualization completed!")
    
    def _create_precision_recall_curves(self, pl_module, trainer):
        """Create precision-recall curves for all labels with global AUC average"""
        if not self.test_prediction_probs or not self.test_true_labels:
            print("No test prediction probabilities available for PR curves")
            return
        
        from sklearn.metrics import precision_recall_curve, auc, average_precision_score
        
        # Concatenate all prediction probabilities and labels
        y_probs = np.concatenate(self.test_prediction_probs, axis=0)
        y_true = np.concatenate(self.test_true_labels, axis=0)
        
        num_classes = y_true.shape[1]
        
        # Get COCO fine categories for labels
        coco = COCO()
        fine_categories = coco.all_fine_categories
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors for different classes
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_classes)))
        if num_classes > 20:
            # For more than 20 classes, use a continuous colormap
            colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
        
        auc_scores = []
        
        # Plot PR curve for each class
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            auc_score = auc(recall, precision)
            avg_precision = average_precision_score(y_true[:, i], y_probs[:, i])
            auc_scores.append(auc_score)
            
            class_name = fine_categories[i] if i < len(fine_categories) else f'Class {i}'
            
            # Only show labels for first few classes to avoid clutter
            if i < 10:
                label = f'{class_name[:10]}... (AUC={auc_score:.3f})' if len(class_name) > 10 else f'{class_name} (AUC={auc_score:.3f})'
                ax.plot(recall, precision, color=colors[i % len(colors)], 
                       label=label, alpha=0.7, linewidth=1.5)
            else:
                ax.plot(recall, precision, color=colors[i % len(colors)], 
                       alpha=0.3, linewidth=0.8)
        
        # Calculate and display global average AUC
        global_avg_auc = np.mean(auc_scores)
        
        # Add random classifier line
        ax.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.5, label='Random Classifier')
        
        # Customize plot
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curves for All Classes\nGlobal Average AUC: {global_avg_auc:.4f}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Add legend (only for first 10 classes to avoid clutter)
        if num_classes <= 10:
            ax.legend(loc='lower left', fontsize=10)
        else:
            # Create custom legend showing only global average
            from matplotlib.lines import Line2D
            custom_lines = [
                Line2D([0], [0], color='gray', alpha=0.7, linewidth=2),
                Line2D([0], [0], color='black', linestyle='--', alpha=0.5)
            ]
            ax.legend(custom_lines, 
                     [f'All Classes (Avg AUC: {global_avg_auc:.4f})', 'Random Classifier'],
                     loc='lower left', fontsize=10)
        
        # Add text box with summary statistics
        textstr = f'''Summary Statistics:
Total Classes: {num_classes}
Global Avg AUC: {global_avg_auc:.4f}
Min AUC: {np.min(auc_scores):.4f}
Max AUC: {np.max(auc_scores):.4f}
Std AUC: {np.std(auc_scores):.4f}'''
        
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"precision_recall_curves": wandb.Image(fig)})
        
        plt.close()
        
        # Create a separate plot showing AUC distribution
        self._create_auc_distribution_plot(auc_scores, fine_categories, trainer)
        
        print(f"Precision-Recall curves completed! Global Average AUC: {global_avg_auc:.4f}")
    
    def _create_auc_distribution_plot(self, auc_scores, fine_categories, trainer):
        """Create a histogram of AUC scores distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Histogram of AUC scores
        ax1.hist(auc_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(auc_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(auc_scores):.4f}')
        ax1.axvline(np.median(auc_scores), color='green', linestyle='--', 
                   label=f'Median: {np.median(auc_scores):.4f}')
        ax1.set_xlabel('AUC Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of AUC Scores Across Classes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Top 10 and Bottom 10 classes by AUC
        sorted_indices = np.argsort(auc_scores)
        bottom_10 = sorted_indices[:10]
        top_10 = sorted_indices[-10:]
        
        # Combine top and bottom
        selected_indices = np.concatenate([bottom_10, top_10])
        selected_aucs = [auc_scores[i] for i in selected_indices]
        selected_names = [fine_categories[i] if i < len(fine_categories) else f'Class {i}' 
                         for i in selected_indices]
        
        # Truncate long names
        selected_names = [name[:15] + '...' if len(name) > 15 else name for name in selected_names]
        
        # Create colors (red for bottom, green for top)
        colors = ['red'] * 10 + ['green'] * 10
        
        y_pos = np.arange(len(selected_names))
        bars = ax2.barh(y_pos, selected_aucs, color=colors, alpha=0.7)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(selected_names, fontsize=8)
        ax2.set_xlabel('AUC Score')
        ax2.set_title('Top 10 and Bottom 10 Classes by AUC (Sorted by Performance)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, auc) in enumerate(zip(bars, selected_aucs)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{auc:.3f}', va='center', fontsize=8)
        
        # Add dividing line between bottom and top
        ax2.axhline(y=9.5, color='black', linestyle='-', alpha=0.5)
        ax2.text(0.5, 9.5, 'Bottom 10 | Top 10', transform=ax2.get_yaxis_transform(), 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Log to wandb
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"auc_distribution": wandb.Image(fig)})
        
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
    
    def on_test_end(self, trainer, pl_module):
        """Create all test visualizations after test phase"""
        self._create_confusion_matrix_plot(pl_module, trainer)
        self._create_classification_report_plot(pl_module, trainer)
        self._create_precision_recall_curves(pl_module, trainer)