# CIFAR-100 Image Classification dengan ConvNeXt V2 & SAM

Proyek ini mengimplementasikan pipeline klasifikasi gambar untuk dataset CIFAR-100 menggunakan model ConvNeXt V2 (melalui `timm`). Pelatihan dioptimalkan menggunakan *Sharpness-Aware Minimization* (SAM) dan dikelola dengan PyTorch Lightning. Kode ini mendukung pelatihan Single-GPU dan Multi-GPU (DDP).

## Fitur Utama

* **Model**: ConvNeXt V2 Tiny (pretrained pada ImageNet-22k dan ImageNet-1k).
* **Optimizer**: SAM (Sharpness-Aware Minimization) untuk generalisasi yang lebih baik.
* **Engine**: PyTorch Lightning.
* **Akselerasi**:
  * Mendukung Multi-GPU (Strategy: DDP) dan Single-GPU secara otomatis.
* **Tuning**: Hyperparameter tuning menggunakan Optuna.
* **Augmentasi**: RandAugment, RandomHorizontalFlip, Mixup (jika diaktifkan di model/loss).

## Struktur Direktori

Pastikan struktur folder Anda seperti berikut agar import berjalan lancar:

```text
.
├── data/                   # Folder dataset (.npz files)
    ├── train.npz
    ├── test.npz          
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Class Dataset dan Transformasi
│   ├── model.py            # LightningModule (Model + Training Logic)
│   └── utils.py            # Implementasi SAM Optimizer
├── train.py                # Script utama pelatihan
├── tune.py                 # Script hyperparameter tuning
└── requirements.txt
```

## Instalasi

Install library yang dibutuhkan:

`pip install torch torchvision pytorch-lightning timm optuna pandas`

## Penggunaan

### 1. Pelatihan Model (Train)

Script `train.py` akan otomatis mendeteksi jumlah GPU. Jika tersedia lebih dari 1 GPU, strategi DDP dan SyncBatchNorm akan aktif.

**Perintah Dasar:**

`python train.py --epochs 20 --batch_size 32 --lr 5e-5`

**Kustomisasi Penuh:**

```
python train.py
    --model_name convnextv2_tiny.fcmae_ft_in22k_in1k
    --epochs 30
    --batch_size 64
    --lr 0.0005
    --weight_decay 0.05
    --rho 0.05
    --accum_steps 1
    --img_size 224
```

### 2. Hyperparameter Tuning (Optuna)

Jalankan `tune.py` untuk mencari parameter terbaik (`lr`, `weight_decay`, `rho`).

`python tune.py --n_trials 50 --storage sqlite:///db.sqlite3`

## Argumen

Berikut adalah daftar argumen yang dapat digunakan pada `train.py`:

| **Argumen**  | **Default**      | **Deskripsi**                           |
| ------------------ | ---------------------- | --------------------------------------------- |
| `--model_name`   | `convnextv2_tiny...` | Nama model dari library `timm`              |
| `--img_size`     | `224`                | Ukuran input gambar                           |
| `--batch_size`   | `32`                 | Ukuran batch per GPU                          |
| `--epochs`       | `20`                 | Jumlah epoch pelatihan                        |
| `--lr`           | `5e-5`               | Learning rate awal                            |
| `--weight_decay` | `0.05`               | Weight decay untuk optimizer                  |
| `--rho`          | `0.05`               | Parameter rho untuk SAM (radius neighborhood) |
| `--accum_steps`  | `1`                  | Gradient accumulation steps                   |
| `--num_workers`  | `2`                  | Jumlah worker dataloader                      |
| `--seed`         | `42`                 | Seed untuk reproduktifitas                    |
| `--data_path`    | `data/`              | Path ke folder dataset                        |
