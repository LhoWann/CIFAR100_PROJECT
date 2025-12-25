import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.dataset import Cifar100NPZDataset, get_transforms
from src.model import LitModel

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default='convnextv2_tiny.fcmae_ft_in22k_in1k')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_path', type=str, default='data/')
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--accum_steps', type=int, default=42)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    
    train_transform, test_transform = get_transforms(args.img_size)
    
    train_path = f"{args.data_path}/train_data.npz"
    test_path = f"{args.data_path}/test_data.npz"

    full_dataset = Cifar100NPZDataset(train_path, is_train=True, transform=train_transform)
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    model = LitModel(
        model_name=args.model_name,
        num_classes=100,
        lr=args.lr,
        weight_decay=args.weight_decay,
        rho=args.rho,
        img_size=args.img_size
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename=f"{args.model_name}-best",
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        accumulate_grad_batches=args.accum_steps,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')]
    )

    trainer.fit(model, train_loader, val_loader)

    if trainer.global_rank == 0:
        test_dataset = Cifar100NPZDataset(test_path, is_train=False, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers)

        best_model = LitModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        best_model.eval()
        best_model.cuda()

        all_ids, all_preds = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                imgs, ids = batch
                imgs = imgs.cuda()
                
                logits = best_model(imgs)
                logits_flipped = best_model(torch.flip(imgs, dims=[3]))
                final_logits = (logits + logits_flipped) / 2.0
                preds = torch.argmax(final_logits, dim=1)
                
                all_ids.extend(ids.numpy())
                all_preds.extend(preds.cpu().numpy())

        sub_name = f"submission_{args.model_name}_{args.img_size}_seed{args.seed}.csv"
        pd.DataFrame({'ID': all_ids, 'Label': all_preds}).to_csv(sub_name, index=False)

if __name__ == "__main__":
    main()