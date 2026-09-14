"""
Losses para treino de SOD: BCE + IoU (padrão na literatura)
"""
import torch
import torch.nn.functional as F


def bce_iou_loss(pred, target):
    """BCE + IoU loss — padrão em modelos SOD."""
    pred   = pred.sigmoid()
    bce    = F.binary_cross_entropy(pred, target, reduction='mean')

    inter  = (pred * target).sum(dim=(1, 2, 3))
    union  = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou    = 1 - (inter + 1) / (union + 1)

    return bce + iou.mean()


def multi_scale_loss(outputs, target):
    """Loss com deep supervision (múltiplas saídas)."""
    if isinstance(outputs, (list, tuple)):
        loss = sum(bce_iou_loss(o, target) for o in outputs)
        return loss / len(outputs)
    return bce_iou_loss(outputs, target)
