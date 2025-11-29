# trainer.py
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from timeit import default_timer as timer
from utils import compute_metrics, save_metrics_csv, save_best_model

def train_model(model, train_loader, val_loader, fold, device, args, class_weights=None):

    num_epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    out_dir = args.out_dir

    print(f"\nStarting training for Fold {fold}...\n")

    # Loss function
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_macro_f1 = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - Fold {fold}")
        start_time = timer()

        # Training Loop
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        epoch_val_loss, metrics = validate_model(model, val_loader, criterion, device,
                                                 out_dir=out_dir, fold=fold, epoch=epoch+1)
        
        # Save Best Model
        best_macro_f1, saved = save_best_model(model, metrics, best_macro_f1, save_dir=out_dir, fold=fold)

        epoch_time = timer() - start_time
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Macro F1: {metrics['macro_f1']:.4f} | Best Macro F1: {best_macro_f1:.4f} | Time: {epoch_time:.2f}s")

    print(f"\nTraining finished for Fold {fold}. Best Macro F1: {best_macro_f1:.4f}\n")

def validate_model(model, val_loader, criterion, device, out_dir=None, fold=None, epoch=None):
 
    #Validate the model and compute metrics.
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)

    # Compute metrics
    class_names = val_loader.dataset.classes
    metrics = compute_metrics(
        y_true, y_pred,
        class_names=class_names,
        save_dir=out_dir,
        fold=fold,
        epoch=epoch
    )
    if out_dir:
        save_metrics_csv(metrics, class_names=class_names, save_dir=out_dir, fold=fold, epoch=epoch)

    return epoch_val_loss, metrics