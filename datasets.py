import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class SkinDataset(Dataset):
    def __init__(self, df, img_dir="images", fold=None, is_train=True, transform=None):
        """
        df: DataFrame with columns: image_id, dx, fold
        img_dir: folder containing all images
        fold: int, used to select train/val for cross-validation
        is_train: True if training, False if validation
        transform: torchvision transforms to apply
        """
        if fold is not None:
            if is_train: # uses all rows not in this fold
                self.df = df[df["fold"] != fold].reset_index(drop=True)
            else: # uses all rows in this fold
                self.df = df[df["fold"] == fold].reset_index(drop=True)
        else: # If no fold is provided, use all data.
            self.df = df.copy()

        # Image directory
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        if img_dir is None:
            self.img_dir = os.path.join(BASE_DIR, "data/images")
        else:
            self.img_dir = img_dir

        self.is_train = is_train
        self.transform = transform

        # Map class names to integer labels ( 0-6)
        self.classes = sorted(self.df["dx"].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.df["label_idx"] = self.df["dx"].map(self.class_to_idx)

        # Default transforms
        if self.transform is None:
            if self.is_train:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(45),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
            else:  # only resizing and normalizing no augmentation
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['image_id']}.jpg"
        image = Image.open(img_path).convert("RGB")
        label = row["label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label
