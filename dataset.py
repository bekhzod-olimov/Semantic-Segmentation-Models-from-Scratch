# Import libraries
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    
    """
    
    This class extracts data from pre-defined directory.
    
    Arguments:
    
        root            - directory path to the dataset, str;
        transformations - transformations to be applied to the data, bool;
        im_files        - image file types, list.
        
    Output:
        
        dataset (with applied transformations).
    
    """
    
    def __init__(self, root, transformations = None, im_files = ['.jpg', '.png', '.jpeg']):
        
        super().__init__()

        self.im_paths = sorted(glob(f"{root}/original_images/*[{im_file for im_file in im_files}]"))
        self.gt_paths = sorted(glob(f"{root}/label_images_semantic/*[{im_file for im_file in im_files}]"))
        self.transformations = transformations
        
    def __len__(self): return len(self.im_paths)
        
    def __getitem__(self, idx):
        
        im, gt = Image.open(self.im_paths[idx]), Image.open(self.gt_paths[idx])
        
        if self.transformations is not None: im, gt = self.transformations(im), self.transformations(gt)
        
        return im, gt
    
def get_dl(root, transformations, bs, split = [0.8, 0.2]):
        
    assert sum(split) == 1., "Sum of the split must be equal to 1"
    
    ds = CustomDataset(root, transformations)
    
    tr_ds, val_ds = torch.utils.data.random_split(ds, split)
    
    print(f"\nThere are {len(tr_ds)} number of images in the train set")
    print(f"There are {len(val_ds)} number of images in the validation set\n")
    
    tr_dl  = DataLoader(tr_ds, batch_size = bs, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = bs, shuffle = False)
    
    
    
    return tr_dl, val_dl
    
# ds = CustomDataset("data/dataset/semantic_drone_dataset")
# print(ds[0])
