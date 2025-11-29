# data_preparation.py
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

def prepare_data(csv_path="~/theme3/skin_cancer_project/data/metadata.csv", n_splits=5, random_state=42):
    """
    Prepare HAM10000 dataset for 5-fold cross-validation.
    - Adds a 'fold' column for CV splits
    - Computes class weights for imbalanced dataset
    """

    # Load metadata
    df = pd.read_csv(csv_path)
    
    # Add fold column
    df["fold"] = -1  # initialized to -1

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["dx"])):
        df.loc[val_idx, "fold"] = fold

    # Save fold assignments
    df.to_csv("~/theme3/skin_cancer_project/data/HAM10000_5folds.csv", index=False)
    print("Saved 5-fold assignments to HAM10000_5folds.csv")

    # Compute class weights
    classes = np.unique(df["dx"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=df["dx"]
    )

    class_weights_dict = dict(zip(classes, class_weights))
    print("Class weights:", class_weights_dict)

    return df, class_weights_dict

if __name__ == "__main__":
    df, class_weights = prepare_data()
