import os

import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune

from utils.parser import parse_args_test_norm_pruning
from utils.utils import get_loader
from utils.prune import one_shot_pruning
from trainer import train
from trainer import eval
from trainer import train_model
  

def main():

    args = parse_args_test_norm_pruning()
    print('------ Parameters for test_norm_pruning ------')
    for parameter, value in args.__dict__.items():
        print(f'{parameter}: {value}')
    print('------------------------------------------')

    ### Get the model, train it if none was given
    if args.model_path is None:
        model = train_model(args)
    else:
        model = torch.load(args.model_path)
    
    ### Save the trained model to make sure to have the same model before pruning.
    try:
        os.mkdir("temp")
    except FileExistsError:
        pass
    torch.save(model, "temp/model_norm_pruning.pt")

    ### Get the loaders 
    if not args.download and args.data_dir == '../data':
        raise("ERROR: please provide the data directory from which to take the data.")
        
    kwargs = {'num_workers': 1, 'pin_memory': True} if (torch.cuda.is_available() and args.use_cuda) else {}
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

    loader_class = get_loader(args.dataset)
    loader_object = loader_class(args.data_dir, args.batch_size, args.test_batch_size, 
                                 args.custom_transforms, args.crop_size)

    loader_train = loader_object.get_loader(train=True, download=args.download, kwargs=kwargs)
    loader_eval = loader_object.get_loader(train=False, download=args.download, kwargs=kwargs)

    ### Testing all the combination between the methods and pruning_rates given
    pruning_rates = args.pruning_rates
    methods = args.pruning_methods

    baseline_accuracy = eval(model, loader_eval, device, args.verbose)

    for method in methods:
        
        accs = []
        
        for pruning_rate in pruning_rates:
            model_ = torch.load("temp/model_norm_pruning.pt")
            accs.append(one_shot_pruning(model_, pruning_rate, loader_train, loader_eval, args.n_epochs_retrain, 
                                         device, args.optimizer, args.loss, args.lr, args.verbose, baseline_accuracy, 
                                         args.save_to, args.show_plot, method))

        plt.plot(pruning_rates, accs, label='Accuracy')
        plt.title('Accuracy w.r.t pruning rate ' + method)
        plt.xlabel('Pruning rate')
        plt.ylabel('Accuracy')
        plt.show()

    
if __name__ == '__main__':
    main()