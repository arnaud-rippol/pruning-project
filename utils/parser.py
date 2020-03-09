import sys
import argparse

import torch


def parse_args_sparsity():

    parser = argparse.ArgumentParser(description='Argument parser for test_sparsity.py')

    parser.add_argument('--size', default=1000, type=int,
                        help='size of the matrix to be multiplied')

    parser.add_argument('--iterations', default=10, type=int,
                        help='number of iterations for each sparsity level')

    parser.add_argument('--sparsities', default=[.95, .98, .99, .995, .997, .999], type=float,
                        nargs="*", help='the levels of sparsities to test')

    parser.add_argument('--verbose', default=False, type=bool,
                        help='print the progress of the sparsity_test')

    parser.add_argument('--show_plot', default=True, type=bool,
                        help='show the plot of the sparsity_test')

    return parser.parse_args()

def parse_args_test_norm_pruning():

    parser = argparse.ArgumentParser(description='Argument parser for test_norm_pruning.py')

    parser.add_argument('--model', default='lenet', type=str,
                        help='the model network to be used')

    parser.add_argument('--model_path', default=None, type=str,
                        help='the path to the model, if saved')                        

    parser.add_argument('--dataset', default='mnist', type=str,
                        help='the name of dataset to use for training')

    parser.add_argument('--verbose', default=False, type=bool,
                        help='print the progress of the sparsity_test')

    parser.add_argument('--show_plot', default=True, type=bool,
                        help='show the plot of the sparsity_test')

    parser.add_argument('--n_epochs', default=15, type=int,
                        help='number of epochs for the training')
    
    parser.add_argument('--download', default=True, type=bool,
                        help='download the data from the torch dataset')

    parser.add_argument('--data_dir', default='../data', type=str,
                        help='The directory of the data. Leave "../data" in case data needs to be downloaded')

    parser.add_argument('--batch_size', default=50, type=int,
                        help='size of the training batch')

    parser.add_argument('--test_batch_size', default=1, type=int,
                        help='size of the testing batch')

    parser.add_argument('--crop_size', default=None, type=int,
                        help='the size to crop the data')

    parser.add_argument('--custom_transforms', default=None, type=str,
                        nargs="*", help='custom transforms to apply to the dataset')

    parser.add_argument('--n_epochs_retrain', default=2, type=int,
                        help='number of epochs for the retraining after the pruning')

    parser.add_argument('--use_cuda', default=True, type=bool,
                        help='use GPU when available')

    parser.add_argument('--save_model', default=True, type=bool,
                        help='save all the epochs of the model to a directory. \
                            To save in a custom directory, use "--save_to".')

    parser.add_argument('--save_to', default=None, type=str,
                        help='the directory in which save the trained model.')
    
    parser.add_argument('--lr', default=0.1, type=float,
                        help='the learning rate to train the model.')

    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='the optimizer to train the model.')

    parser.add_argument('--loss', default='mse', type=str,
                        help='the loss to train the model.')
    
    parser.add_argument('--pruning_rates', default=[0.3, 0.6, 0.8, 0.9, 0.93, 0.95, 0.97], type=float,
                        nargs="*", help='the levels of pruning to test')
    
    parser.add_argument('--pruning_methods', default=["random", "l1_unstructured","l1_structured", "l2_structured"], 
                        type=str, nargs="*", help='the pruning methods, to chose among "random", "l1_unstructured", \
                            "l1_structured" and "l2_structured")')


    return parser.parse_args()


def parse_args_test_gradual_pruning():

    parser = argparse.ArgumentParser(description='Argument parser for test_norm_pruning.py')

    parser.add_argument('--model', default='lenet', type=str,
                        help='the model network to be used')

    parser.add_argument('--model_path', default=None, type=str,
                        help='the path to the model, if saved')                        

    parser.add_argument('--dataset', default='mnist', type=str,
                        help='the name of dataset to use for training')

    parser.add_argument('--verbose', default=False, type=bool,
                        help='print the progress of the sparsity_test')

    parser.add_argument('--show_plot', default=True, type=bool,
                        help='show the plot of the sparsity_test')

    parser.add_argument('--n_epochs', default=2, type=int,
                        help='number of epochs for the training')
    
    parser.add_argument('--download', default=True, type=bool,
                        help='download the data from the torch dataset')

    parser.add_argument('--data_dir', default='../data', type=str,
                        help='The directory of the data. Leave "../data" in case data needs to be downloaded')

    parser.add_argument('--batch_size', default=50, type=int,
                        help='size of the training batch')

    parser.add_argument('--test_batch_size', default=1, type=int,
                        help='size of the testing batch')

    parser.add_argument('--crop_size', default=None, type=int,
                        help='the size to crop the data')

    parser.add_argument('--custom_transforms', default=None, type=str,
                        nargs="*", help='custom transforms to apply to the dataset')

    parser.add_argument('--use_cuda', default=True, type=bool,
                        help='use GPU when available')

    parser.add_argument('--n_epochs_retrain', default=2, type=int,
                        help='number of epochs for the retraining after the pruning')

    parser.add_argument('--save_model', default=True, type=bool,
                        help='save all the epochs of the model to a directory. \
                            To save in a custom directory, use "--save_to".')

    parser.add_argument('--save_to', default=None, type=str,
                        help='the directory in which save the trained model.')
    
    parser.add_argument('--lr', default=0.1, type=float,
                        help='the learning rate to train the model.')

    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='the optimizer to train the model.')

    parser.add_argument('--loss', default='mse', type=str,
                        help='the loss to train the model.')
    
    parser.add_argument('--final_sparsity', default=0.9, type=float,
                        nargs="*", help='the levels of pruning to test')

    parser.add_argument('--pruning_epochs', default=2, type=int,
                        help='number of epochs for the training')

    parser.add_argument('--pruning_frequency', default=1, type=int,
                        help='prune the model every "pruning_frequency" epoch during the retraining')


    parser.add_argument('--pruning_method', default="l1_unstructured", type=str, 
                        help='the pruning method, to chose among "random", "l1_unstructured", \
                            "l1_structured" and "l2_structured")')


    return parser.parse_args()
