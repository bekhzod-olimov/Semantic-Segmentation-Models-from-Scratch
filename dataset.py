import torch, cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from glob import glob
from PIL import ImageFile
from transformations import get_transformations
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    
    def __init__(self, root, transformations = None, im_files = ['.jpg', '.png', '.jpeg']):
        
        super().__init__()

        self.im_paths = sorted(glob(f"{root}/original_images/*[{im_file for im_file in im_files}]"))
        self.gt_paths = sorted(glob(f"{root}/label_images_semantic/*[{im_file for im_file in im_files}]"))
        self.transformations = transformations
        self.tensorize = T.Compose([T.ToTensor()])
        
    def __len__(self): return len(self.im_paths)
        
    def __getitem__(self, idx):
        
        im, gt = cv2.cvtColor(cv2.imread(self.im_paths[idx]), cv2.COLOR_BGR2RGB), cv2.imread(self.gt_paths[idx], 0)
        if self.transformations is not None: 
            transformed = self.transformations(image = im, mask = gt)
            im, gt = transformed['image'], transformed['mask']
        return self.tensorize(im), torch.tensor(gt).long()
    
def get_dl(root, transformations, bs, split=[0.85, 0.15]):
        
    assert sum(split) == 1., "Sum of the split must be equal to 1"
    
    ds = CustomDataset(root, transformations)
    tr_ds, val_ds = torch.utils.data.random_split(ds, split)
        
    print(f"\nThere are {len(tr_ds)} number of images in the train set")
    print(f"There are {len(val_ds)} number of images in the validation set\n")
    
    tr_dl  = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = 8)
    val_dl = DataLoader(val_ds, batch_size = bs, shuffle = False, num_workers = 8)
    
    return tr_dl, val_dl
    
# ds = CustomDataset("data/dataset/semantic_drone_dataset")
# print(ds[0])
