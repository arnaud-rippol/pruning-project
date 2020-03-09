import os
import time

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt

from trainer import train
from trainer import eval
from trainer import train_model 
from utils.parser import parse_args_test_gradual_pruning
from utils.utils import get_loader
from utils.prune import one_shot_pruning
from utils.prune import gradual_linear_pruning
from utils.prune import automated_gradual_pruning


def main():

    args = parse_args_test_gradual_pruning()
    print('------ Parameters for test_gradual_pruning ------')
    for parameter, value in args.__dict__.items():
        print(f'{parameter}: {value}')
    print('------------------------------------------')

    if args.model_path is None:
        if args.verbose:
            print(f"No model was given, training {args.model} on {args.dataset} with {args.n_epochs} epochs.")
        model = train_model(args)
    else:
        model = torch.load(args.model_path)

    try:
        os.mkdir("temp")
    except FileExistsError:
        pass
    torch.save(model, "temp/model_gradual_pruning.pt")


    if not args.download and args.data_dir == '../data':
        raise("ERROR: please provide the data directory from which to take the data.")

    kwargs = {'num_workers': 1, 'pin_memory': True} if (torch.cuda.is_available() and args.use_cuda) else {}
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

    loader_class = get_loader(args.dataset)
    loader_object = loader_class(args.data_dir, args.batch_size, args.test_batch_size, 
                                 args.custom_transforms, args.crop_size)

    loader_train = loader_object.get_loader(train=True, download=args.download, kwargs=kwargs)
    loader_eval = loader_object.get_loader(train=False, download=args.download, kwargs=kwargs)


    baseline_accuracy = eval(model, loader_eval, device, args.verbose)

    results_one_shot = one_shot_pruning(model, args.final_sparsity, loader_train, loader_eval, args.n_epochs_retrain, 
                                        device, args.optimizer, args.loss, args.lr, args.verbose, baseline_accuracy, 
                                        args.save_to, args.show_plot, args.pruning_method)  
    if args.verbose:
        print(f"Accuracy obtained with one-shot pruning: {results_one_shot}") 

    results_linear_pruning = gradual_linear_pruning(model, args.final_sparsity, loader_train, loader_eval, 
                                                    args.n_epochs_retrain, args.pruning_epochs, args.pruning_frequency, 
                                                    device, args.optimizer, args.loss, args.lr, args.verbose, 
                                                    baseline_accuracy, args.save_to, args.show_plot, args.pruning_method)
    if args.verbose:
        print(f"Accuracy obtained with linear gradual pruning: {results_one_shot}") 

    results_AGP = gradual_linear_pruning(model, args.final_sparsity, loader_train, loader_eval, args.n_epochs_retrain,
                                         args.pruning_epochs, args.pruning_frequency, device, args.optimizer,
                                         args.loss, args.lr, args.verbose, baseline_accuracy, args.save_to, 
                                         args.show_plot, args.pruning_method)
    if args.verbose:
        print(f"Accuracy obtained with automated gradual pruning: {results_AGP}") 


if __name__ == '__main__':
    main()