
def dice(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    return ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)).mean()

def dice_loss(pred, target, smooth = 1.):
    return 1 - dice(pred, target, smooth)

def iou(pred, target, smooth=1e-4):
    pred = pred.contiguous()
    target = target.contiguous() 
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)
    union=union+smooth
    union[union==0]=smooth
    return ((intersection + smooth) / union).mean()
    

def iou_loss(pred, target, smooth = 1e-4):
    return 1 - iou(pred,target,smooth)
    

    
def tversky_index(pred, target, alpha , beta,smooth = 1e-4):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)

    X_Y = pred.sum(dim=2).sum(dim=2) - intersection
    Y_X = target.sum(dim=2).sum(dim=2) - intersection
    loss = (1 - (intersection + smooth)/ ((intersection + smooth) + alpha * X_Y + beta * Y_X))
    
    return loss.mean()

