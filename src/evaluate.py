import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import get_model
from datasets import SkinDataset
from utils import save_confusion_matrix
from args import get_args


def evaluate(model, dataloader, device, class_names, save_dir):
    """
    Runs evaluation on unseen dataset.

    Args:
        model: trained PyTorch model
        dataloader: test DataLoader
        device: cuda/cpu
        class_names: list of class labels
        save_dir: where to save metrics + confusion matrix
    """

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    report = classification_report(all_labels, all_preds, target_names=class_names)

    # Save metrics
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "evaluation_metrics.txt"), "w") as f:
        f.write("=== Evaluation Metrics ===\n\n")
        f.write(report)
        f.write(f"\nMacro F1 Score: {macro_f1:.4f}\n")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    save_confusion_matrix(cm, class_names, os.path.join(save_dir, "confusion_matrix.png"))

    print("\nEvaluation complete!")
    print(report)
    print(f"Macro F1: {macro_f1:.4f}")


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOAD TEST CSV
    test_csv = os.path.join(args.data_dir, "test.csv")  # YOU MUST PROVIDE THIS
    df = pd.read_csv(test_csv)

    class_names = sorted(df["label"].unique())

    # Dataset + Loader
    test_dataset = SkinDataset(df, args.data_dir, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Load model
    model = get_model(backbone=args.backbone, num_classes=len(class_names))
    model_path = os.path.join(args.out_dir, "best_model.pth")  # YOU MUST SAVE THIS DURING TRAINING
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluate
    evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        save_dir=os.path.join(args.out_dir, "test_results")
    )


if __name__ == "__main__":
    main()
