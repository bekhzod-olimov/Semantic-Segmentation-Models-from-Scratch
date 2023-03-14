import time, torch, os
from tqdm import tqdm
from metrics import Metrics
import numpy as np
import torch.nn.functional as F


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def tic_toc(start_time = None): return time.time() if start_time == None else time.time() - start_time
    
def train(model, tr_dl, val_dl, loss_fn, opt, sch, device, epochs, save_prefix):
    
    tr_loss, tr_pa, tr_iou = [], [], []
    val_loss, val_pa, val_iou = [], [], []
    tr_len, val_len = len(tr_dl), len(val_dl)
    best_loss, decrease, not_improve = np.inf, 1, 0
    os.makedirs('saved_models', exist_ok=True)

    model.to(device)
    train_start = tic_toc()
    print("Starting train process...")
    
    for epoch in range(1, epochs + 1):
        tic = tic_toc()
        tr_loss_, tr_iou_, tr_pa_ = 0, 0, 0
        
        model.train()
        print(f"Epoch {epoch} train is started...")
        
        for idx, batch in enumerate(tqdm(tr_dl)):
            
            im, gt = batch
            im, gt = im.to(device), gt.flatten(0,1).type(torch.LongTensor).to(device)
            
            pred = model(im)
            met = Metrics(pred, gt, loss_fn)
            loss_ = met.loss()
            
            tr_iou_ += met.mIoU()
            print(f"My mIoU: {met.mIoU()}")
            print(f"Original mIoU: {mIoU(pred, gt)}")
            tr_pa_ += met.PA()
            print(f"PA: {met.PA()}")
            print(f"Original PA: {pixel_accuracy(pred, gt)}")
            tr_loss_ += loss_.item()
            
            loss_.backward()
            opt.step()
            opt.zero_grad()
            sch.step()
            
        print(f"Epoch {epoch} train is finished.")
        
        print(f"Epoch {epoch} validation is started...")
        model.eval()
        val_loss_, val_iou_, val_pa_ = 0, 0, 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_dl)):

                im, gt = batch
                im, gt = im.to(device), gt.flatten(0,1).type(torch.LongTensor).to(device)

                pred = model(im)
                met = Metrics(pred, gt, loss_fn)

                val_iou_ += met.mIoU()
                val_pa_ += met.PA()
                val_loss_ += met.loss().item()

        print(f"Epoch {epoch} validation is finished.")

        tr_loss_ /= tr_len
        tr_iou_ /= tr_len
        tr_pa_ /= tr_len

        val_loss_ /= val_len
        val_iou_ /=  val_len
        val_pa_ /=   val_len

        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"\nEpoch {epoch} Train Process Results: \n")
        print(f"Train Time      -> {tic_toc(tic):.3f} secs")
        print(f"Train Loss      -> {tr_loss_:.3f}")
        print(f"Train PA        -> {tr_pa_:.3f}")
        print(f"Train IoU       -> {tr_iou_:.3f}")
        print(f"Validation Loss -> {val_loss_:.3f}")
        print(f"Validation PA   -> {val_pa_:.3f}")
        print(f"Validation IoU  -> {val_iou_:.3f}\n")

        tr_loss.append(tr_loss_)
        tr_iou.append(tr_iou_)
        tr_pa.append(tr_pa_)

        val_loss.append(val_loss_)
        val_iou.append(val_iou_)
        val_pa.append(val_pa_)
        
        if best_loss > (val_loss_):
            print(f'Loss decreased from {best_loss:.3f} to {val_loss_:.3f}')
            best_loss = val_loss_
            decrease += 1
            if decrease % 2 == 0:
                print('Saving the best model with the lowest loss value...')
                torch.save(model, f'saved_models/{save_prefix}_best_model_{val_loss_:.3f}.pt')

        if val_loss_ > best_loss:

            not_improve += 1
            best_loss = val_loss_
            print(f'Loss value did not decrease for {not_improve} epochs')
            if not_improve == 7:
                print('Loss value did not decrease for 7 epochs; Stop Training...')
                break
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
            
    print(f"Train is completed in {(tic_toc(start_train)) / 60:.3f} mins")
    
    return {"tr_loss": tr_loss, "tr_iou": tr_iou, "tr_pa": tr_pa,
            "val_loss": val_loss, "val_iou": val_iou, "val_pa" : val_pa}
            
            
            
            
            
            
            
        
        
    
    
    
