import argparse
import os

def get_args():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Skin Lesion Classification")

    parser.add_argument("-data_dir", type=str, default=os.path.join(BASE_DIR, "data/images"))
                        
    parser.add_argument("-csv_dir", type=str, default=os.path.join(BASE_DIR, "data/HAM10000_5folds.csv"))
                        
    parser.add_argument("-out_dir", type=str, default=os.path.join(BASE_DIR, "outputs"))

    parser.add_argument("-n_splits", type=int, default=5, help="Number of folds for cross-validation")
                        
    # Training parameters

    parser.add_argument("-epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("-batch_size", type=int, default=32,
                        choices = [16, 32, 64])
    parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate")

    parser.add_argument("-weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")

    parser.add_argument("-num_workers", type=int, default=4, choices = [4, 6, 8], 
                        help="Number of DataLoader workers")
    
    parser.add_argument("-seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("-backbone", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("-pretrained", type=bool, default=True, help="Use pretrained weights or not")

    args = parser.parse_args()
    return args
