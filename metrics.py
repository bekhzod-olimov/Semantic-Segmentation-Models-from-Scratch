# Import libraries
import torch
import torch.nn.functional as F
import numpy as np

class Metrics():
    
    """
    
    This class gets predicted and ground truth masks, loss function, epsilon value, and number of classes;
    and computes pixel accuracy (PA), mean intersection over union (mIoU), and loss values.
    
    Arguments:
    
        pred    - predicted mask, tensor;
        gt      - ground truth mask, tensor;
        loss_fn - loss function, torch loss function;
        eps     - epsilon value, float;
        n_cls   - number of classes, int.
        
    Outputs:
    
        PA      - pixel accuracy value, float;
        mIoU    - mean intersection over union value, float;
        loss    - loss value, float.
    
    """
    
    def __init__(self, pred, gt, loss_fn, eps = 1e-10, n_cls = 23):
        
        self.pred, self.gt = torch.argmax(F.softmax(pred, dim=1), dim=1), gt
        self.loss_fn, self.eps, self.n_cls, self.pred_ = loss_fn, eps, n_cls, pred
        
    def to_contiguous(self, inp): return inp.contiguous().view(-1)
    
    def PA(self):

        with torch.no_grad():
            
            match = torch.eq(self.pred, self.gt).int()
        
        return float(match.sum()) / float(match.numel())

    def mIoU(self):
        
        with torch.no_grad():
            
            pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)

            iou_per_class = []
            
            for c in range(self.n_cls):
                
                match_pred = pred == c
                match_gt   = gt == c

                if match_gt.long().sum().item() == 0: iou_per_class.append(np.nan)
                    
                else:
                    
                    intersect = torch.logical_and(match_pred, match_gt).sum().float().item()
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()

                    iou = (intersect + self.eps) / (union + self.eps)
                    iou_per_class.append(iou)
                    
            return np.nanmean(iou_per_class)
    
    def loss(self): 
        return self.loss_fn(self.pred_, self.gt)

# t = get_transformations()[1]
# loss_fn = torch.nn.CrossEntropyLoss()
# pred = t(Image.open('data/dataset/semantic_drone_dataset/original_images/566.jpg'))
# gt = t(Image.open('data/dataset/semantic_drone_dataset/original_images/566.jpg'))
# met = Metrics(pred, gt, loss_fn)
# print(met.PA())
# print(met.mIoU())
# print(met.loss())
