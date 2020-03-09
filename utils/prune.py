import numpy as np
import torch
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt

from trainer import train
from trainer import eval
from trainer import train_model 


def prune_model(method_name, parameters_to_prune, pruning_rate):
    if method_name == 'l1_unstructured':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_rate,
        )
    
    elif method_name == 'random':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=pruning_rate,
        )
    
    elif method_name == 'l1_structured':
        for (module, name) in parameters_to_prune:
            prune.ln_structured(module=module, name=name, n=2, amount=pruning_rate, dim=-1)
            prune.ln_structured(module=module, name=name, n=2, amount=pruning_rate, dim=-1)
            prune.ln_structured(module=module, name=name, n=2, amount=pruning_rate, dim=-1)
    
    elif method_name == 'l2_structured':
        for (module, name) in parameters_to_prune:
            prune.ln_structured(module=module, name=name, n=2, amount=pruning_rate, dim=-1)
            prune.ln_structured(module=module, name=name, n=2, amount=pruning_rate, dim=-1)
            prune.ln_structured(module=module, name=name, n=2, amount=pruning_rate, dim=-1)
    
    else:
        raise("Pruning method not found")


def one_shot_pruning(model, final_sparsity, train_loader, test_loader, n_epochs, device, optimizer, loss, lr, verbose,
                     acc_baseline, save_path, show_plot, pruning_method):
    
    if verbose:
        print(f"------One shot pruning, global sparsity: {100*final_sparsity} % ------")
    
    parameters_to_prune = [(module, 'weight') for module in model.modules()][1:]

    prune_model(pruning_method, parameters_to_prune, final_sparsity)

    sparsity_list = np.zeros(n_epochs + 1)
    accuracy_list = np.zeros(n_epochs + 1)   
    sparsity_list[0] = 0    
    accuracy_list[0] = acc_baseline 
     
    for epoch in range(1, n_epochs + 1):                       
        if verbose:
            print(f"Retraining: epoch {epoch} / {n_epochs}")
        train(model, train_loader, device, optimizer, loss, lr, verbose)
        acc = eval(model, test_loader, device, verbose)
        sparsity_list[epoch] = final_sparsity
        accuracy_list[epoch] = acc

    if save_path is not None:
        torch.save(model, f"{save_path}_one_shot_pruning.pt")

    if show_plot:
        plt.plot(np.arange(n_epochs + 1), sparsity_list, label='Sparsity')
        plt.plot(np.arange(n_epochs + 1), accuracy_list, label='Accuracy')
        plt.title('Fine-tuning curve')
        plt.xlabel('Epochs')
        plt.ylabel('Sparsity / Accuracy')
        plt.legend(loc="lower left")
        plt.show()
    
    return acc 


def gradual_linear_pruning(model, final_sparsity, train_loader, test_loader, n_epochs, pruning_epochs, frequency, 
                           device, optimizer, loss, lr, verbose, acc_baseline, save_path, show_plot, pruning_method):
    
    if verbose:
        print(f"------Gradual linear pruning, global sparsity: {100*final_sparsity} % ------")

    parameters_to_prune = [(module, 'weight') for module in model.modules()][1:]
    
    sparsity_list = np.zeros(n_epochs + 1)
    accuracy_list = np.zeros(n_epochs + 1)
    sparsity_list[0] = 0
    accuracy_list[0] = acc_baseline   

    sparsity = 0
    nb_of_pruning = 0

    for epoch in range(1, n_epochs + 1):
        if verbose:
            print(f"Retraining: epoch {epoch} / {n_epochs}")

        if nb_of_pruning < pruning_epochs and (epoch - 1) % frequency == 0:      
            nb_of_pruning += 1        
            
            new_sparsity = final_sparsity * nb_of_pruning / pruning_epochs
            pruning_rate = (new_sparsity - sparsity) / (1 - sparsity)
            
            prune_model(pruning_method, parameters_to_prune, pruning_rate)
            
            sparsity = new_sparsity                                
        train(model, train_loader, device, optimizer, loss, lr, verbose)
        acc = eval(model, test_loader, device, verbose)
        sparsity_list[epoch] = sparsity
        accuracy_list[epoch] = acc
        print("Global sparsity:", sparsity)

    if save_path is not None:
        torch.save(model, f"{save_path}_gradual_linear_pruning.pt")

    if show_plot:
        plt.plot(np.arange(n_epochs + 1), sparsity_list, label='Sparsity')
        plt.plot(np.arange(n_epochs + 1), accuracy_list, label='Accuracy')
        plt.title('Fine-tuning curve')
        plt.xlabel('Epochs')
        plt.ylabel('Sparsity / Accuracy')
        plt.legend(loc="lower left")
        plt.show()
    
    return acc 


def automated_gradual_pruning(model, final_sparsity, train_loader, test_loader, n_epochs, pruning_epochs, frequency, 
                              device, optimizer, loss, lr, verbose, acc_baseline, save_path, show_plot, pruning_method):

    if verbose:
        print(f"------Automated gradual pruning, global sparsity: {100*final_sparsity} % ------")

    sparsity = 0
    nb_of_pruning = 0
    parameters_to_prune = [(module, 'weight') for module in model.modules()][1:]

    sparsity_list = np.zeros(n_epochs + 1)
    accuracy_list = np.zeros(n_epochs + 1)   
    sparsity_list[0] = 0    
    accuracy_list[0] = acc_baseline

    for epoch in range(1, n_epochs + 1):
        if verbose:
            print(f"Retraining: epoch {epoch / {n_epochs}}")
        if nb_of_pruning < pruning_epochs and (epoch - 1) % frequency == 0:
            nb_of_pruning += 1    
            new_sparsity = final_sparsity * (1 - (1 - nb_of_pruning / pruning_epochs) ** 3)
            pruning_rate = (new_sparsity - sparsity) / (1 - sparsity)
            
            prune_model(pruning_method, parameters_to_prune, pruning_rate)

            sparsity = new_sparsity                                          
        train(model, train_loader, device, optimizer, loss, lr, verbose)
        acc = eval(model, test_loader, device, verbose)
        sparsity_list[epoch] = sparsity
        accuracy_list[epoch] = acc
        print("Global sparsity:", sparsity)
      
    if save_path is not None:
        torch.save(model, f"{save_path}_gradual_linear_pruning.pt")

    if show_plot:
        plt.plot(np.arange(n_epochs + 1), sparsity_list, label='Sparsity')
        plt.plot(np.arange(n_epochs + 1), accuracy_list, label='Accuracy')
        plt.title('Fine-tuning curve')
        plt.xlabel('Epochs')
        plt.ylabel('Sparsity / Accuracy')
        plt.legend(loc="lower left")
        plt.show()
    
    return acc 

    