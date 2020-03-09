import os

import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune

from utils.parser import parse_args_test_norm_pruning
from utils.utils import get_loader
from trainer import train
from trainer import eval
from trainer import train_model

def train_eval(model, train_loader, test_loader, pruning_rate, method, n_epochs, device, optimizer_name, loss_name, lr):
    
    parameters_to_prune = [(module, 'weight') for module in model.modules()][1:]
    
    if method == "random":
        #Random
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=pruning_rate,
        )

    if method == "l1_unstructured":
        #L1
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_rate,
        )
    
    if method == "l2_structured":
        #Ln
        prune.ln_structured(module=model.fc1, name='weight', n=2, amount=pruning_rate, dim=-1)
        prune.ln_structured(module=model.fc2, name='weight', n=2, amount=pruning_rate, dim=-1)
        prune.ln_structured(module=model.fc3, name='weight', n=2, amount=pruning_rate, dim=-1)
    
    if method == "l1_structured":
        #Ln
        prune.ln_structured(module=model.fc1, name='weight', n=1, amount=pruning_rate, dim=-1)
        prune.ln_structured(module=model.fc2, name='weight', n=1, amount=pruning_rate, dim=-1)
        prune.ln_structured(module=model.fc3, name='weight', n=1, amount=pruning_rate, dim=-1)

       
    for epoch in range(1, n_epochs + 1):
        print("Epoch n°", epoch, ":")
        train(model, train_loader, device, optimizer_name, loss_name, lr)
        acc = eval(model, test_loader, device)
        print(" ")

    # Saving pruned LeNet-5
    name = "lenet/lenet_pruning_rate_" + str(pruning_rate) + method + ".pt"
    torch.save(model, name)
    
    return acc
  

def main():

    args = parse_args_test_norm_pruning()
    print('------ Parameters for test_sparsity ------')
    for parameter, value in args.__dict__.items():
        print(f'{parameter}: {value}')
    print('------------------------------------------')

    if args.model_path is None:
        model = train_model(args)
    else:
        model = torch.load(args.model_path)
    
    try:
        os.mkdir("temp")
    except FileExistsError:
        pass
    torch.save(model, "temp/model_norm_pruning.pt")

    pruning_rates = args.pruning_rates
    methods = args.pruning_methods

    if not args.download and args.data_dir == '../data':
        print("ERROR: please provide the data directory from which to take the data.")
        

    kwargs = {'num_workers': 1, 'pin_memory': True} if (torch.cuda.is_available() and args.use_cuda) else {}
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

    loader_class = get_loader(args.dataset)
    loader_object = loader_class(args.data_dir, args.batch_size, args.test_batch_size, 
                                 args.custom_transforms, args.crop_size)

    loader_train = loader_object.get_loader(train=True, download=args.download, kwargs=kwargs)
    loader_eval = loader_object.get_loader(train=False, download=args.download, kwargs=kwargs)


    for method in methods:
        
        accs = []
        
        for pruning_rate in pruning_rates:
            model_ = torch.load("temp/model_norm_pruning.pt")
            accs.append(train_eval(model_, loader_train, loader_eval, pruning_rate, method, args.n_epochs, device,
                        args.optimizer, args.loss, args.lr))

        plt.plot(pruning_rates, accs, label='Accuracy')
        plt.title('Accuracy w.r.t pruning rate ' + method)
        plt.xlabel('Pruning rate')
        plt.ylabel('Accuracy')
        plt.show()

    
if __name__ == '__main__':
    main()