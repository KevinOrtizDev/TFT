import torch
import torch.nn.functional as F

def dice(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    return ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)).mean()

def dice_loss(pred, target, smooth = 1.):
    # pred = pred.contiguous()
    # target = target.contiguous()    

    # intersection = (pred * target).sum(dim=2).sum(dim=2)

    # loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return 1 - dice(pred, target, smooth)

def iou(pred, target, smooth=1e-4):
    pred = pred.contiguous()
    target = target.contiguous() 
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)
    union[union==-smooth]=smooth
    return ((intersection + smooth) / (union + smooth)).mean()
    

def iou_loss(pred, target, smooth = 1e-4):
    # pred = pred.contiguous()
    # target = target.contiguous()    

    # intersection = (pred * target).sum(dim=2).sum(dim=2)
    # union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)
      
    return 1 - iou(pred,target,smooth)
    

    
def tversky_index(pred, target, alpha , beta,smooth = 1e-4):
    pred = pred.contiguous()
    target = target.contiguous()    
    print(alpha)
    print(beta)
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)

    X_Y = pred.sum(dim=2).sum(dim=2) - intersection
    Y_X = target.sum(dim=2).sum(dim=2) - intersection
    loss = (1 - (intersection + smooth)/ ((intersection + smooth) + alpha * X_Y + beta * Y_X))
    
    return loss.mean()

def calc_loss_dice(pred, target, metrics, bce_weight=0.1):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)
    
    #loss = bce * bce_weight + dice * (1 - bce_weight)
    loss = bce * bce_weight + dice
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def calc_loss_iou(pred, target, metrics, bce_weight=0.1):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    iou = iou_loss(pred, target)
    
    #loss = bce * bce_weight + dice * (1 - bce_weight)
    loss = bce * bce_weight + iou
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['iou'] += iou.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def calc_loss_tversky(pred, target, metrics, bce_weight=0.1):
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = torch.sigmoid(pred)
    # Alpha, beta values (https://arxiv.org/pdf/1810.07842v1.pdf)
    alpha=.7
    beta=.3
    tversky = tversky_index(pred, target, alpha , beta)
    
    #loss = bce * bce_weight + dice * (1 - bce_weight)
    loss = bce * bce_weight + tversky
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['tversky'] += tversky.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

    
def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))