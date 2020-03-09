import os

import torch
import matplotlib.pyplot as plt
import numpy as np

from trainer import train_model
from trainer import eval
from utils.parser import parse_args_finetuning_pruning
from utils.utils import get_loader
from utils.prune import gradual_linear_pruning


def main():

    args = parse_args_finetuning_pruning()
    print('------ Parameters for finetuning ------')
    for parameter, value in args.__dict__.items():
        print(f'{parameter}: {value}')
    print('---------------------------------------')

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
    torch.save(model, "temp/model_finetuning_parameters.pt")


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
    accuracy_list = [baseline_accuracy]

    n_epochs_retrain = args.n_epochs_retrain

    for n_pruning_epochs in range(1, n_epochs_retrain + 1):
        model_ = torch.load("temp/model_finetuning_parameters.pt")
        accuracy_list.append(gradual_linear_pruning(model_, args.final_sparsity, loader_train, loader_eval, 
                                                    n_epochs_retrain, n_pruning_epochs, 1, device, args.optimizer,
                                                    args.loss, args.lr, args.verbose, baseline_accuracy, args.save_to, 
                                                    False, args.pruning_method))

    if args.show_plot:
        plt.plot(np.arange(n_epochs_retrain + 1), accuracy_list, label='Accuracy')
        plt.xlabel('Pruning rate')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower left")
        plt.show()


if __name__ == '__main__':
    main()