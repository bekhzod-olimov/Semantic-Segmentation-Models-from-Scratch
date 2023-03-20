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

        # Get paths of the images and masks 
        self.im_paths = sorted(glob(f"{root}/original_images/*[{im_file for im_file in im_files}]"))
        self.gt_paths = sorted(glob(f"{root}/label_images_semantic/*[{im_file for im_file in im_files}]"))
        
        # Set transformations
        self.transformations = transformations
        
    def __len__(self): return len(self.im_paths)
        
    def __getitem__(self, idx):
        
        # Get the pair of images and masks in the specific index
        im, gt = Image.open(self.im_paths[idx]), Image.open(self.gt_paths[idx])
        
        # Apply transformations if exist
        if self.transformations is not None: im, gt = self.transformations(im), self.transformations(gt)
        
        return im, gt
    
def get_dl(root, transformations, bs, split = [0.8, 0.2]):
    
    """
    
    This function gets path of the data, transformations, batch size, and split options; creates dataloaders and return train and validation dataloaders.
    
    Arguments:
    
        root            - directory path to the dataset, str;
        bs              - batch size, int;
        transformations - transformations to be applied to the data, bool;
        split           - split ratios, Python list with ints.
        
    Output:
        
        tr_dl           - train dataloader;
        val_dl          - validation dataloader. 
    
    """
        
    # Make sure sum of the split list equals to 1 
    assert sum(split) == 1., "Sum of the split must be equal to 1"
    
    # Get the dataset
    ds = CustomDataset(root, transformations)
    
    # Split the dataset into train and validation sets
    tr_ds, val_ds = torch.utils.data.random_split(ds, split)
    
    print(f"\nThere are {len(tr_ds)} number of images in the train set")
    print(f"There are {len(val_ds)} number of images in the validation set\n")
    
    # Create train and validation dataloaders
    tr_dl  = DataLoader(tr_ds,  batch_size = bs, shuffle = True)
    val_dl = DataLoader(val_ds, batch_size = bs, shuffle = False)
    
    return tr_dl, val_dl
    
# ds = CustomDataset("data/dataset/semantic_drone_dataset")
# print(ds[0])
