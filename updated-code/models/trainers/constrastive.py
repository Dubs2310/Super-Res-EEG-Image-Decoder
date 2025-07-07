import wandb 
import torch
import random
from torch.optim import Adam
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from utils.singletons.coco import COCO
from models.definers.encoder import EEGEncoderDefiner as EEGEncoder
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score

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
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_accuracy = MultilabelAccuracy(num_labels=num_fine_labels)
        self.val_accuracy = MultilabelAccuracy(num_labels=num_fine_labels)
        self.v2_acc = MultilabelAccuracy(num_labels=2)
        self.v4_acc = MultilabelAccuracy(num_labels=4)
        self.v10_acc = MultilabelAccuracy(num_labels=10)
        self.v80_acc = MultilabelAccuracy(num_labels=80)
        self.top5_acc = MeanMetric()
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
        return self.validation_step(batch, batch_idx)
    

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

# Rest of the classes remain the same
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
    
    def on_train_end(self, trainer, pl_module):
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        
        axs[0, 0].plot(self.train_losses, label='Train Loss')
        axs[0, 0].plot(self.test_losses, label='Val Loss')
        axs[0, 0].legend()
        axs[0, 0].set_title("Loss Curve")
        
        axs[0, 1].plot(self.train_accuracies, label='Train Accuracy')
        axs[0, 1].plot(self.test_accuracies, label='Val Accuracy')
        axs[0, 1].legend()
        axs[0, 1].set_title("Accuracy Curve")
        
        axs[1, 0].plot(self.v2_accs, label='2-class Accuracy')
        axs[1, 0].legend()
        axs[1, 0].set_title("2-Class Accuracy Curve")
        
        axs[1, 1].plot(self.v4_accs, label='4-class Accuracy')
        axs[1, 1].legend()
        axs[1, 1].set_title("4-Class Accuracy Curve")
        
        axs[2, 0].plot(self.v10_accs, label='10-class Accuracy')
        axs[2, 0].legend()
        axs[2, 0].set_title("10-Class Accuracy Curve")

        axs[2, 0].plot(self.v10_accs, label='80-class Accuracy')
        axs[2, 0].legend()
        axs[2, 0].set_title("80-Class Accuracy Curve")
        
        info_text = (f"Best Model Info (from Epoch {pl_module.best_epoch_info.get('epoch', 0)}):\n"
                    f"Train Loss: {pl_module.best_epoch_info.get('train_loss', 0):.4f}\n"
                    f"Train Accuracy: {pl_module.best_epoch_info.get('train_accuracy', 0):.4f}\n"
                    f"Val Loss: {pl_module.best_epoch_info.get('val_loss', 0):.4f}\n"
                    f"Val Accuracy: {pl_module.best_epoch_info.get('val_accuracy', 0):.4f}\n"
                    f"v2_acc:{pl_module.best_epoch_info.get('v2_acc', 0):.4f}\n"
                    f"v4_acc:{pl_module.best_epoch_info.get('v4_acc', 0):.4f}\n"
                    f"v10_acc:{pl_module.best_epoch_info.get('v10_acc', 0):.4f}\n"
                    f"v80_acc:{pl_module.best_epoch_info.get('v80_acc', 0):.4f}")
        
        axs[2, 1].axis('off')
        axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)
        
        plt.tight_layout()
        plt.suptitle('EEG Model Training Results', fontsize=16, y=1.05)
        
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            trainer.logger.experiment.log({"training_plots": wandb.Image(fig)})
        
        plt.savefig('training_results.png')
        plt.close()