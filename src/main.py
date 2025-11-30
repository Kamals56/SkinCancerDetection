import os
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from args import get_args
from datasets import SkinDataset
from models import get_model
from trainer import train_model
from utils import check_device, get_fold_dir


device = check_device()

def main():
    args = get_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Reading the CSV files
    df = pd.read_csv(args.csv_dir)

    for fold in range(args.n_splits):
        print(f"Training fold: {fold}")

        # Create fold-specific folder to save metrices of each fold
        fold_dir = get_fold_dir(args.out_dir, fold)

        # Preparing the datasets
        train_dataset = SkinDataset(df, img_dir=args.data_dir, fold=fold, is_train=True)
        val_dataset = SkinDataset(df, img_dir=args.data_dir, fold=fold, is_train=False)

        # Intitializing the DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2,
            persistent_workers=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2,
            persistent_workers=True,
        )

        # Initializing teh  model
        model = get_model(backbone=args.backbone, 
                          num_classes=7, 
                          pretrained=args.pretrained)
        model = model.to(device)

        # balancing the
        classes = np.unique(df["dx"])
        class_weights = compute_class_weight(class_weight="balanced",
                                     classes=classes,
                                     y=df["dx"])
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # Train the model for this fold
        train_model(model, train_loader, val_loader, fold, device, args, class_weights=class_weights)

        # Cleanup per fold
        del train_loader, val_loader, train_dataset, val_dataset, model
        gc.collect()
        torch.cuda.empty_cache()

    print("All folds finished.")

if __name__ == "__main__":
    main()
