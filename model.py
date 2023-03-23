# Import libraries
import segmentation_models_pytorch as smp
from models.unet import UNet

def get_model(model_name, classes, encoder_depth, model_type): 
    
    """
    
    This function gets name of the model, number of classes, depth of the encoder, and model type and returns initialized model.
    
    Arguments:
    
        model_name - name of the model for training, str;
        classes    - number of classes in the dataset, int;
        
    
    """
    
    if model_type == 'smp':
        print(f"UNet model with pretrained {model_name} backbone weights is successfully loaded!\n")
        return smp.Unet(encoder_name = model_name, classes = classes, encoder_depth = encoder_depth,
                        encoder_weights = 'imagenet', activation = None, decoder_channels = [256, 128, 64, 32, 16])
    
    elif model_type == 'unet':
        print(f"Original UNet without pretrained weights is successfully loaded!\n")
        return UNet(in_chs     = 3,
                    n_cls      = 23,
                    out_chs    = 64,
                    depth      =  4,
                    up_method  =  'bilinear')
