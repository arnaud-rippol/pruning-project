import torch.optim as optim
import torch.nn as nn
from loaders.mnist_loader import MNISTloader
from models.lenet import LeNet


def get_loader(dataset_name):
    if dataset_name == "mnist":
        return MNISTloader


def get_model(model_name):
    if model_name == "lenet":
        return LeNet()


def get_optimizer(optimizer_name):
    if optimizer_name == 'sgd':
        return optim.SGD
    elif optimizer_name == 'ADAM':
        return optim.Adam

def get_loss(loss_name):
    if loss_name == "mse":
        return nn.MSELoss