import torch
from torch.amp import autocast

from .utils import checking

def train_one_epoch(model, criterion, data_loader, optimizer, scaler, device, max_norm=0, k=5):
    running_loss = 0
    total_targets = 0
    correct_predictions_1 = 0
    correct_predictions_k = 0
    model.train()
    criterion.train()
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        with autocast(device_type=device, cache_enabled=True):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            loss = sum(loss_dict.values())
        
        scaler.scale(loss).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if torch.is_tensor(outputs):
            num_targets, num_correct_1, num_correct_k = checking(outputs, targets, k=k)
        else:
            num_targets, num_correct_1, num_correct_k = checking(outputs[0], targets, k=k)
        total_targets += num_targets
        correct_predictions_1 += num_correct_1
        correct_predictions_k += num_correct_k

    metrics = {
        "loss": running_loss / len(data_loader),
        "acc_1": 100 * correct_predictions_1 / total_targets,
        "acc_2": 100 * correct_predictions_k / total_targets,
    }
    return metrics

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, k=5):
    running_loss = 0
    total_targets = 0
    correct_predictions_1 = 0
    correct_predictions_k = 0
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = targets.to(device)

            with autocast(device_type=device, cache_enabled=True):
                outputs = model(samples)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())

            running_loss += loss.item()

            if torch.is_tensor(outputs):
                num_targets, num_correct_1, num_correct_k = checking(outputs, targets, k=k)
            else:
                num_targets, num_correct_1, num_correct_k = checking(outputs[0], targets, k=k)
            total_targets += num_targets
            correct_predictions_1 += num_correct_1
            correct_predictions_k += num_correct_k

    metrics = {
        "loss": running_loss / len(data_loader),
        "acc_1": 100 * correct_predictions_1 / total_targets,
        "acc_2": 100 * correct_predictions_k / total_targets,
    }
    return metrics