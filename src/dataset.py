import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Cifar100NPZDataset(Dataset):
    def __init__(self, npz_path, is_train=True, transform=None):
        data = np.load(npz_path)
        if is_train:
            self.x = data['x_train']
            self.y = data['y_train']
            self.ids = None
        else:
            self.x = data['x_test']
            self.y = None
            self.ids = data['ID']
        
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        img = torch.tensor(img, dtype=torch.uint8)
        
        if self.transform:
            img = self.transform(img)

        if self.is_train:
            label = torch.tensor(self.y[idx], dtype=torch.long)
            return img, label
        else:
            return img, self.ids[idx]

def get_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.RandAugment(num_ops=2, magnitude=14),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, test_transform