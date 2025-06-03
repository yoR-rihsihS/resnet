import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode


@torch.no_grad()
def checking(outputs, targets, k=3):
    """ 
    Computes the number of correct predictions, top-1 and top-k.
    Args:
        - outputs (Tensor): Raw logits of shape [bs, num_classes]
        - targets (Tensor): Integer class labels of shape [bs]
        - k (int): Parameter k for computing top-k accuracy
    Returns:
        - total_targets (int): total number of ground truth targets.
        - correct_predictions_1 (int): total number of (top-1) correct predictions.
        - correct_predictions_5 (int): total number of (top-5) correct predictions.
    """
    total_targets = 0
    correct_predictions_1 = 0
    correct_predictions_k = 0

    predictions = torch.argmax(outputs, dim=1)
    correct_predictions_1 += (predictions == targets).sum().item()
    total_targets += targets.size(0)
    
    topk_preds = outputs.topk(k, dim=1).indices
    targets_expanded = targets.view(-1, 1).expand_as(topk_preds)
    correct_topk = (topk_preds == targets_expanded).any(dim=1)
    correct_predictions_k = correct_topk.sum().item()

    return total_targets, correct_predictions_1, correct_predictions_k

def get_activation(act=None, inpace=True):
    if act == "silu":
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU() 
    elif act == "prelu":
        m = nn.PReLU()
    elif act is None:
        m = nn.Identity()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError(f"Unknown activation {act} requested")  
    if hasattr(m, 'inplace'):
        m.inplace = inpace
    return m

def get_transforms(img_size=224):
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomPhotometricDistort(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=(-0.15, 0.15), p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.7, 1.2), shear=(-15, 15, -15, 15), interpolation=InterpolationMode.BILINEAR),
        v2.RandomChoice([
            v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            v2.RandomCrop((img_size, img_size)),
        ], p=[0.67, 0.33]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.6388, 0.5445, 0.4448], std=[0.2229, 0.2414, 0.2638]),
    ])

    eval_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.6388, 0.5445, 0.4448], std=[0.2229, 0.2414, 0.2638]),
    ])

    return train_transforms, eval_transforms

class FoodX251Dataset(Dataset):
    def __init__(self, images_dir, labels_csv, transform=None):
        """
        Dataset for FoodX-251 with image paths and integer labels from a CSV file.
        Args:
            images_dir (str): Directory containing images.
            labels_csv (str): Path to the CSV file with columns: [image_name, label].
            transform (callable, optional): Image transformations to apply.
        """
        self.images_dir = images_dir
        self.transform = transform

        df = pd.read_csv(labels_csv)
        self.image_names = df['img_name'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_names[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label