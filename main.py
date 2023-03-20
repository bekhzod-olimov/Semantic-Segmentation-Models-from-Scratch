# Import libraries
import argparse, torch
from dataset import get_dl
from transformations import get_transformations
from metrics import Metrics
from model import get_model
from train import train

def run(args):
    
    # Get transformations
    transformations = get_transformations()[0]
    
    # Get train and validation dataloaders
    tr_dl, val_dl = get_dl(args.root, transformations, args.batch_size)
    
    # Get train model    
    model = get_model(args.model_name, args.n_cls, args.depth, args.model_type)
    
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Initialize optimizer to update trainable parameters
    opt = torch.optim.AdamW(model.parameters(), lr = args.learning_rate)
    
    # Initialize scheduler for the optimizer
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.learning_rate, epochs = args.epochs, steps_per_epoch = len(tr_dl))
    
    his = train(model, tr_dl, val_dl, loss_fn, opt, sched, args.device, args.epochs, args.model_type)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Semantic Segmentation Train Arguments")
    
    parser.add_argument("-r", "--root", type = str, default = 'data/dataset/semantic_drone_dataset', help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 6, help = "Mini-batch size")
    parser.add_argument("-mn", "--model_name", type = str, default = 'mobilenet_v2', help = "Model name for backbone")
    parser.add_argument("-mt", "--model_type", type = str, default = 'unet', help = "Name of the semantic segmentation model")
    parser.add_argument("-cl", "--n_cls", type = int, default = 23, help = "Number of classes in the dataset")
    parser.add_argument("-d", "--depth", type = int, default = 5, help = "Depth of the encoder")
    parser.add_argument("-dev", "--device", type = str, default = 'cpu', help = "GPU device number")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 1e-3, help = "Learning rate value")
    parser.add_argument("-e", "--epochs", type = int, default = 30, help = "Train epochs number")
    
    args = parser.parse_args() 
    
    run(args) 
