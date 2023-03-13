from torchvision import transforms as T

def get_transformations():

    return [T.Compose([T.Resize((704, 1056)),
                      T.RandomVerticalFlip(),
                      T.RandomVerticalFlip(),
                      T.GaussianBlur(kernel_size = 3),
                      T.ToTensor()]),
           
            T.Compose([T.Resize((704, 1056)),
                      T.ToTensor()])
           ]
