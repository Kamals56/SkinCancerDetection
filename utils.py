# utils.py
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def check_device():
    """Check if GPU is available, else use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_fold_dir(base_dir, fold):
    """
    Create a folder for each fold inside the base directory.
    Returns the path to that fold's directory.
    """
    fold_dir = os.path.join(base_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    return fold_dir


def compute_metrics(y_true, y_pred, class_names=None, save_dir=None, fold=None, epoch=None):
    """
    Compute metrics for classification and optionally save confusion matrix.

    Args:
        y_true (list or np.array): True labels
        y_pred (list or np.array): Predicted labels
        class_names (list): List of class names
        save_dir (str): Directory to save confusion matrix image
        fold (int): Fold number
        epoch (int): Epoch number

    Returns:
        metrics_dict (dict): Dictionary containing all computed metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Macro metrics
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    metrics_dict = {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm,
        "balanced_accuracy": balanced_acc
    }

    # Save confusion matrix as image
    if save_dir and class_names is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Fold {fold} Epoch {epoch} Confusion Matrix" if fold is not None and epoch is not None else "Confusion Matrix")
        plt.tight_layout()
        file_name = f"confusion_matrix_fold{fold}_epoch{epoch}.png" if fold is not None and epoch is not None else "confusion_matrix.png"
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

    return metrics_dict


def save_metrics_csv(metrics_dict, class_names, save_dir, fold=None, epoch=None):
    """
    Save per-class metrics and macro metrics to a CSV file.

    Args:
        metrics_dict (dict): Output from compute_metrics()
        class_names (list): List of class names
        save_dir (str): Directory to save CSV
        fold (int, optional): Fold number
        epoch (int, optional): Epoch number
    """
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.DataFrame({
        "class": class_names,
        "f1_score": metrics_dict["per_class_f1"],
        "precision": metrics_dict["per_class_precision"],
        "recall": metrics_dict["per_class_recall"]
    })

    macro_row = pd.DataFrame([{
        "class": "macro_avg",
        "f1_score": metrics_dict["macro_f1"],
        "precision": metrics_dict["macro_precision"],
        "recall": metrics_dict["macro_recall"]
    }])
    df = pd.concat([df, macro_row], ignore_index=True)

    file_name = f"metrics_fold{fold}_epoch{epoch}.csv" if fold is not None and epoch is not None else "metrics.csv"
    df.to_csv(os.path.join(save_dir, file_name), index=False)


def save_best_model(model, metrics_dict, best_macro_f1, save_dir, fold=None):
    """
    Save the best model based on Macro F1-score.

    Args:
        model (nn.Module): PyTorch model
        metrics_dict (dict): Output from compute_metrics()
        best_macro_f1 (float): Current best Macro F1
        save_dir (str): Directory to save model
        fold (int, optional): Fold number

    Returns:
        best_macro_f1 (float): Updated best Macro F1
        saved (bool): Whether model was saved
    """
    os.makedirs(save_dir, exist_ok=True)
    saved = False
    current_macro_f1 = metrics_dict["macro_f1"]

    if current_macro_f1 > best_macro_f1:
        best_macro_f1 = current_macro_f1
        model_path = os.path.join(save_dir, f"best_model_fold{fold}.pth" if fold is not None else "best_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model at {model_path} with Macro F1: {best_macro_f1:.4f}")
        saved = True

    return best_macro_f1, saved


def save_loss_curve(train_losses, val_losses, save_dir, fold=None):
    """
    Save training and validation loss curves.

    Args:
        train_losses (list): Training loss per epoch
        val_losses (list): Validation loss per epoch
        save_dir (str): Directory to save curve
        fold (int, optional): Fold number
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve Fold {fold}" if fold is not None else "Loss Curve")
    plt.legend()
    file_name = f"loss_curve_fold{fold}.png" if fold is not None else "loss_curve.png"
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()
