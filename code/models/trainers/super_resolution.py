import torch
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
        self.mean_train_loss = MeanMetric()
        self.mean_val_loss = MeanMetric()
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
        self.mean_train_loss(loss)
        self.log("train_batch_loss", self.mean_train_loss)
        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", self.mean_train_loss)

    def validation_step(self, batch, batch_idx):
        lo_res, hi_res = batch
        lo_res = (lo_res[0] if len(lo_res) == 2 else lo_res).float()
        hi_res = (hi_res[0] if len(hi_res) == 2 else hi_res).float()
        super_res = self.super_resolution_model(lo_res)
        loss = self.super_resolution_model.compute_loss(hi_res, super_res)
        self.mean_val_loss(loss)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_loss", self.mean_val_loss)

    def test_step(self, batch, batch_idx):
        lo_res, hi_res = batch
        lo_res = (lo_res[0] if len(lo_res) == 2 else lo_res).float()
        hi_res = (hi_res[0] if len(hi_res) == 2 else hi_res).float()
        super_res = self.super_resolution_model(lo_res)
        loss = self.super_resolution_model.compute_loss(hi_res, super_res)
        self.mean_test_loss(loss)
        self.mean_test_snr(super_res, hi_res)
        self.mean_test_mae(super_res, hi_res)
        self.mean_test_mse(super_res, hi_res)
        self.mean_test_nrmse(super_res, hi_res)
        self.mean_test_pearson(super_res, hi_res)
        return loss

    def on_test_epoch_end(self):
        self.log("test_loss", self.mean_test_loss)
        self.log("test_snr", self.mean_test_snr)
        self.log("test_mae", self.mean_test_mae)
        self.log("test_mse", self.mean_test_mse)
        self.log("test_nrmse", self.mean_test_nrmse)
        self.log("test_pearson", self.mean_test_pearson)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))
        return optimizer


class SuperResolutionTrainerCallback(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.test_metrics = {}
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss', 0)
        if torch.is_tensor(train_loss):
            train_loss = train_loss.cpu().item()
        self.train_losses.append(train_loss)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss', 0)
        if torch.is_tensor(val_loss):
            val_loss = val_loss.cpu().item()
        self.val_losses.append(val_loss)
    
    def on_test_epoch_end(self, trainer, pl_module):
        test_metrics = ['test_loss', 'test_snr', 'test_mae', 'test_mse', 'test_nrmse', 'test_pearson']
        for metric in test_metrics:
            value = trainer.callback_metrics.get(metric, 0)
            if torch.is_tensor(value):
                value = value.cpu().item()
            self.test_metrics[metric] = value