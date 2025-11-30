HAM10000 Skin Lesion Classification

## Project Overview
This project implements a **deep learning-based skin lesion classification system** using the **HAM10000 dataset**. The model leverages **pretrained ResNet architectures** with **data augmentation** and **cross-validation** to classify 7 types of skin lesions.  

The system includes training, evaluation, and visualization of performance metrics, including confusion matrices, loss curves, and macro F1 scores.  

---

## Features
- Supports **ResNet18, ResNet34, and ResNet50** backbones.
- Handles **class imbalance** using **weighted loss**.
- Implements **5-fold cross-validation**.
- Performs **on-the-fly data augmentation**:
  - RandomResizedCrop  
  - Horizontal/Vertical Flip  
  - Random Rotation  
  - Color Jitter
- Saves:
  - **Best model per fold**  
  - **Loss curves per fold**  
  - **Metrics and confusion matrices per epoch**
- Early stopping and learning rate scheduling support.
- Evaluate on **unseen test dataset**.

---

## Dataset
- Dataset: [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- Classes:

| Index | Name                           | Short Name |
|-------|--------------------------------|------------|
| 0     | Actinic Keratoses              | akiec      |
| 1     | Basal Cell Carcinoma           | bcc        |
| 2     | Benign Keratosis-like Lesions | bkl        |
| 3     | Dermatofibroma                 | df         |
| 4     | Melanocytic Nevi               | nv         |
| 5     | Melanoma                       | mel        |
| 6     | Vascular Lesions               | vasc       |

---

## Installation
```bash
# Clone repository
git clone <repo-url>

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Ensure your data is structured correctly for the system:

* **Image Location:** Place all skin lesion images in a single, dedicated directory (default: `data/images`).
* **Metadata File:** Provide a **CSV file** that maps images to their labels and specifies the cross-validation assignment. This file must contain the columns: `image_id`, `dx` (the class label), and `fold` (for 5-fold cross-validation).

---

### 2. Training (with Cross-Validation)

The `main.py` script initiates the training process, which runs **5-fold cross-validation**. The system trains the model five times, using a different fold as the validation set each time, and saves the best performing model from each fold.

```bash
python main.py \
    -data_dir data/images \
    -csv_dir data/HAM10000_5folds.csv \
    -out_dir session \
    -backbone resnet50 \
    -epochs 50 \
    -batch_size 32 \
    -lr 1e-4 \
    -weight_decay 1e-5 \
    -scheduler plateau \
    -patience 5

```
    


Training uses **cross-validation**, saving the **best model per fold** along with metrics and loss curves.

---

### 3. Evaluation on Test Data

To evaluate the model's performance on data it has not encountered during training, use the `evaluate.py` script. This computes metrics and the final confusion matrix on **unseen data**.

```bash
python evaluate.py \
    -model_dir session/fold_0/best_model.pth \
    -test_csv data/test.csv \
    -test_dir data/test_images
```

Computes metrics and confusion matrix on unseen data.

## Outputs

* **`session/fold_{i}/best_model.pth`**: **Best model for fold i.**

* **`session/fold_{i}/metrics_fold{i}_epoch{e}.csv`**: → **Metrics per epoch.**

* **`session/fold_{i}/confusion_matrix_fold{i}_epoch{e}.png`**: → **Confusion matrices.**

* **`session/fold_{i}/loss_curve_fold{i}.png`**: → **Training and validation loss curves.**

## Results

* Average Macro F1 score: 0.81

* Confusion matrices and metrics per fold can be visualized for analysis.

## Acknowledgements

* Dataset: HAM10000: Tschandl et al., 2018

* PyTorch: Deep learning framework

* Torchvision: Pretrained models and image transforms

## License

This project is licensed under the MIT License.