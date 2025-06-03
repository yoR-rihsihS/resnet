import os
import argparse
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.amp import GradScaler

from resnet import get_ResNet, train_one_epoch, evaluate, Criterion, FoodX251Dataset, get_transforms

DEVICE = "cuda"
torch.cuda.empty_cache()

def save_file(history, path):
    with open(path, 'wb') as file:
        pickle.dump(history, file)

def load_file(path):
    with open(path, 'rb') as file:
        history = pickle.load(file)
    return history

def print_metrics(metrics, epoch, mode, k):
    print(f"Epoch {epoch} - {mode} Metrics:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy: {metrics['acc_1']:.4f}")
    print(f"  Top-{k} Accuracy: {metrics['acc_2']:.4f}")

def main(cfg):
    model = get_ResNet(name=cfg["model_name"], num_classes=cfg["num_classes"], aux_loss=cfg["aux_loss"])
    model.to(DEVICE)

    criterion = Criterion(
        weight_dict=cfg["weight_dict"],
        losses=cfg["compute_losses"],
        num_classes=cfg["num_classes"],
    )

    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of parameters =", num_parameters)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters =", num_parameters)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step"], gamma=cfg['gamma'])
    scaler = GradScaler()

    train_transform, eval_transform = get_transforms()
    train_set = FoodX251Dataset(images_dir=cfg["train_path"][0], labels_csv=cfg["train_path"][1], transform=train_transform)
    val_set = FoodX251Dataset(images_dir=cfg["val_path"][0], labels_csv=cfg["val_path"][1], transform=eval_transform)
    
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=5, prefetch_factor=10, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, prefetch_factor=10, persistent_workers=True, pin_memory=True)

    history = {"train": [], "val": []}

    if cfg["aux_loss"]:
        cfg["model_name"] += "_aux"

    if os.path.exists(f"./saved/{cfg['model_name']}_checkpoint.pth"):
        history = load_file(f"./saved/{cfg['model_name']}_history.pkl")
        checkpoint = torch.load(f"./saved/{cfg['model_name']}_checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    for epoch in range(cfg['epochs']):
        if len(history['train']) > epoch:
            print_metrics(history['train'][epoch], epoch+1, "Train", cfg['k'])
            print_metrics(history['val'][epoch], epoch+1, "Validation", cfg['k'])
            print()
            continue

        train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, scaler, DEVICE, max_norm=0.1, k=cfg['k'])
        print_metrics(train_metrics, epoch+1, "Train", cfg['k'])
        val_metrics = evaluate(model, criterion, val_loader, DEVICE, k=cfg['k'])
        print_metrics(val_metrics, epoch+1, "Validation", cfg['k'])

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print()
        scheduler.step()

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, f"./saved/{cfg['model_name']}_checkpoint.pth")
        save_file(history, f"./saved/{cfg['model_name']}_history.pkl")

    torch.save(model.state_dict(), f"./saved/{cfg['model_name']}_{epoch+1}.pth")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="ResNet Training")
    parser.add_argument("--model", type=str, required=True, help="Model name to train")
    parser.add_argument("--aux_loss", type=bool, required=False, default=False, help="Set auxilary heads and loss for training")
    args = parser.parse_args()
    model_name = args.model
    aux_loss = args.aux_loss
    config = {
        "model_name": model_name,
        "num_classes": 251,
        "aux_loss": aux_loss,

        "weight_dict": {
            "ce_loss": 0.3,
            "focal_loss": 0.7,
        },
        "compute_losses": ["ce_loss", "focal_loss"],

        "learning_rate": 0.001,
        "epochs": 100,
        "weight_decay": 0.001,
        "gamma": 0.5,
        "step": 15,
        "batch_size": 128,

        "train_path": ["./data/train_set/", "./data/train_labels.csv"],
        "val_path": ["./data/val_set/", "./data/val_labels.csv"],
        "test_path": ["./data/test_set/", "./data/test_labels.csv"],

        "k": 3,
    }
    main(config)