# Import libraries
import torch, cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from glob import glob
from PIL import ImageFile
from transformations import get_transformations
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    
    """
    
    This class gets path to data, transformations, image files and returns images and masks for the specific index.
    
    Arguments:
    
        root            - path to the directory with data, str;
        transformations - transforms to be applied to the output images, albumentations object;
        im_files        - names of the image files to be returned, list.
    
    """
    
    def __init__(self, root, transformations = None, im_files = ['.jpg', '.png', '.jpeg']):
        
        super().__init__()

        # Get images paths
        self.im_paths = sorted(glob(f"{root}/original_images/*[{im_file for im_file in im_files}]"))
        
        # Get masks paths
        self.gt_paths = sorted(glob(f"{root}/label_images_semantic/*[{im_file for im_file in im_files}]"))
        
        # Initialize transformations
        self.transformations = transformations
        
        # Transform array to tensor
        self.tensorize = T.Compose([T.ToTensor()])
        
    def __len__(self): return len(self.im_paths)
        
    def __getitem__(self, idx):
        
        """
    
        This function gets index and returns corresponding image and mask from the dataset. 

        Arguments:

            idx            - index of the data to be returned, int;

        """
        
        # Get an image and corresponding mask (0 for grayscale mask)
        im, gt = cv2.cvtColor(cv2.imread(self.im_paths[idx]), cv2.COLOR_BGR2RGB), cv2.imread(self.gt_paths[idx], 0)
        
        # Apply transformations
        if self.transformations is not None: 
            transformed = self.transformations(image = im, mask = gt)
            
        # Return the transformed image and mask
        return self.tensorize(transformed['image']), torch.tensor(transformed['mask']).long()
    
def get_dl(root, transformations, bs, split=[0.85, 0.15]):
    
    """
    
    This function gets root, transformations, bs, and split and returns train and validation dataloaders.
    
    Arguments:
    
        root            - path to the data, str;
        transformations - transforms to be applied to the data, torchvision transforms object;
        bs              - mini batch size, int;
        split           - split ratios, list -> int.
        
    Outputs:
        
        tr_dl           - train dataloader;
        val_dl          - validation dataloader.
        
    """
        
    assert sum(split) == 1., "Sum of the split must be equal to 1"
    
    # Get dataset
    ds = CustomDataset(root, transformations)
    
    # Split the dataset based on the pre-defined split ratios
    tr_ds, val_ds = torch.utils.data.random_split(ds, split)
        
    print(f"\nThere are {len(tr_ds)} number of images in the train set")
    print(f"There are {len(val_ds)} number of images in the validation set\n")
    
    return DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = 8), DataLoader(val_ds, batch_size = bs, shuffle = False, num_workers = 8)
    
# ds = CustomDataset("data/dataset/semantic_drone_dataset")
# print(ds[0])
