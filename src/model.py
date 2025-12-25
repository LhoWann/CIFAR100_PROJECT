import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from .utils import SAM

class LitModel(pl.LightningModule):
    def __init__(self, model_name, num_classes, lr, weight_decay, rho, img_size):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False 
        
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_path_rate=0.2
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        x, y = batch
        
        logits = self(x)
        loss = self.criterion(logits, y)
        self.manual_backward(loss)
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        opt.first_step(zero_grad=True)
        
        logits_2 = self(x)
        loss_2 = self.criterion(logits_2, y)
        self.manual_backward(loss_2)
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        opt.second_step(zero_grad=True)
        
        sch.step()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            self.parameters(),
            base_optimizer,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            rho=self.hparams.rho
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]