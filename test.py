import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datasets import SkinDataset  # your dataset class

# --------------------------
# Config
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, "data/images")
metadata_csv = os.path.join(BASE_DIR, "data/metadata.csv")

# Load metadata
df = pd.read_csv(metadata_csv)

# Pick 5 random samples
random_rows = df.sample(5).reset_index(drop=True)

# Create dataset with default transforms
dataset = SkinDataset(df, img_dir=img_dir, fold=None, is_train=True)

# --------------------------
# Function to plot images
# --------------------------
def show_image(img, title=None):
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

# --------------------------
# Display original and transformed images
# --------------------------
for idx, row in random_rows.iterrows():
    img_path = os.path.join(img_dir, row["image_id"] + ".jpg")
    img = Image.open(img_path).convert("RGB")
    print(f"Original image: {row['image_id']}, Label: {row['dx']}")
    show_image(img, title=f"Original: {row['dx']}")

    # Get transformed image from dataset
    # Find index in dataset
    ds_idx = dataset.df.index[dataset.df["image_id"] == row["image_id"]][0]
    img_tensor, label_idx = dataset[ds_idx]

    # Convert tensor back to image for plotting
    img_trans = img_tensor.permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
    # Unnormalize for display
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_trans = img_trans * std + mean
    img_trans = img_trans.clip(0, 1)

    show_image(img_trans, title=f"Transformed: {row['dx']}")
