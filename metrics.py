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
        
        # Get predicted and ground truth masks for evaluation
        self.pred, self.gt = torch.argmax(F.softmax(pred, dim = 1), dim = 1), gt
        
        # Get loss function, epsilon value, number of classes, and original predicted mask (for loss value calculation) 
        self.loss_fn, self.eps, self.n_cls, self.pred_ = loss_fn, eps, n_cls, pred
        
    # Move to contiguous
    def to_contiguous(self, inp): return inp.contiguous().view(-1)
    
    def PA(self):
        
        """
        
        This function computes pixel accuracy between predicted and ground truth masks.
        
        Output:
        
            pixel accuracy score.
        
        """

        with torch.no_grad():
            
            # Get number of matching pixels
            match = torch.eq(self.pred, self.gt).int()
        
        # Compute pixel accuracy
        return float(match.sum()) / float(match.numel())

    def mIoU(self):
        
        """
        
        This function computes mean intersection over union between predicted and ground truth masks.
        
        Output:
        
            mean intersection over union value.
        
        """
        
        with torch.no_grad():
            
            # Change predicted and ground truth masks to contiguous values
            pred, gt = self.to_contiguous(self.pred), self.to_contiguous(self.gt)

            # Initialize a list to compute iou for each class in the dataset
            iou_per_class = []

            # Go through each class in the dataset
            for c in range(self.n_cls):
                
                # Get number of prediction pixels for a specific class
                match_pred = pred == c
                
                # Get number of ground truth pixels for a specific class
                match_gt   = gt == c
                
                # In case gt image has no pixels for the class
                if match_gt.long().sum().item() == 0: iou_per_class.append(np.nan)
                    
                else:
                    
                    # Compute intersection
                    intersect = torch.logical_and(match_pred, match_gt).sum().float().item()
                    
                    # Compute union
                    union = torch.logical_or(match_pred, match_gt).sum().float().item()

                    # Comput iou
                    iou = (intersect + self.eps) / (union + self.eps)
                    
                    # Append to the list
                    iou_per_class.append(iou)
            
            # Return mean intersection over union value
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
