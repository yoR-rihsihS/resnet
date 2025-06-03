import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.ops import sigmoid_focal_loss

class Criterion(nn.Module):
    """ This class computes the loss for Image Classification
    """
    def __init__(self, weight_dict, losses, num_classes, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        """ Creates the criterion.
        Args:
            - weight_dict (dict): Contains as key the names of the losses and as values their relative weight.
            - losses [List]: Losses to be applied. Available losses are cross entropy and focal loss.
            - num_classes (int): Number of classes.
            - alpha (float): Focal loss parameter.
            - gamma (float): Focal loss parameter.
            - label_smoothing: Value of epsilon for label smoothing in Cross Entropy.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def ce_loss(self, outputs, targets):
        """
        Computes the cross entropy loss
        """
        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(outputs)
                true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            loss = torch.sum(-true_dist * F.log_softmax(outputs, dim=1), dim=1).mean()
        else:
            loss = F.cross_entropy(outputs, targets, reduction='mean')
        return {"ce_loss": loss}

    def focal_loss(self, outputs, targets):
        """
        Computes the focal loss
        """
        one_hot_targets = F.one_hot(targets, num_classes=self.num_classes).float()
        loss = sigmoid_focal_loss(outputs, one_hot_targets, self.alpha, self.gamma, reduction='mean')
        return {"focal_loss": loss}

    def get_loss(self, loss_name, outputs, targets):
        function_map = {
            "ce_loss": self.ce_loss,
            "focal_loss": self.focal_loss,
        }
        if loss_name not in function_map:
            raise ValueError(f"Unknown loss '{loss_name}'")
        return function_map[loss_name](outputs, targets)

    def forward(self, outputs, targets):
        """
        Compute requested losses and weigh them
        Args:
            - outputs (Tensor | (List[Tensor])): Raw logits of shape [bs, num_classes] or List of Raw logits
            - targets (Tensor): Integer class labels of shape [bs]
        Returns:
            - losses (dict): {loss_name: weighted_loss_value}
        """
        losses = {}
        for loss in self.losses:
            if torch.is_tensor(outputs):
                l_dict = self.get_loss(loss, outputs, targets)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                losses.update(l_dict)
            else:
                for i, raw_logits in enumerate(outputs):
                    l_dict = self.get_loss(loss, raw_logits, targets)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    if i > 0:
                        l_dict = {f"{k}_aux_{i+1}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses