# Import library
from torchvision import transforms as T

def get_transformations():
    
    """
    
    This function intiializes train and validation set transformations.
    
    
    Output:
    
        transformations   -  a list of transformations for train and validation, respectively, list.
        
    """

    return [T.Compose([T.Resize((704, 1056)),
                      T.RandomVerticalFlip(),
                      T.RandomVerticalFlip(),
                      T.GaussianBlur(kernel_size = 3),
                      T.ToTensor()]),
           
            T.Compose([T.Resize((704, 1056)),
                      T.ToTensor()])
           ]
