import argparse
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from src.dataset import Cifar100NPZDataset, get_transforms
from src.model import LitModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--storage', type=str, default='sqlite:///db.sqlite3')
    return parser.parse_args()

class Objective:
    def __init__(self, data_path):
        self.data_path = data_path
        self.img_size = 224
        self.batch_size = 16
        
        train_transform, _ = get_transforms(self.img_size)
        full_path = f"{data_path}/train_data.npz"
        
        self.full_dataset = Cifar100NPZDataset(full_path, is_train=True, transform=train_transform)
        train_len = int(0.9 * len(self.full_dataset))
        val_len = len(self.full_dataset) - train_len
        self.train_set, self.val_set = random_split(self.full_dataset, [train_len, val_len])

    def __call__(self, trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        rho = trial.suggest_float("rho", 0.01, 0.1)

        model = LitModel(
            model_name='convnextv2_tiny.fcmae_ft_in22k_in1k',
            num_classes=100,
            lr=lr,
            weight_decay=weight_decay,
            rho=rho,
            img_size=self.img_size
        )

        train_loader = DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True, 
            drop_last=True
        )
        val_loader = DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )

        n_gpus = torch.cuda.device_count()
        strategy = 'ddp' if n_gpus > 1 else 'auto'
        sync_bn = True if n_gpus > 1 else False

        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="gpu",
            devices=n_gpus,
            strategy=strategy,
            precision="16-mixed",
            sync_batchnorm=sync_bn,
            enable_checkpointing=False,
            logger=False,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")]
        )

        trainer.fit(model, train_loader, val_loader)
        
        return trainer.callback_metrics["val_acc"].item()

if __name__ == "__main__":
    args = parse_args()
    objective = Objective(args.data_path)
    
    study = optuna.create_study(
        direction="maximize", 
        storage=args.storage, 
        study_name="cifar100_convnext_tuning",
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")