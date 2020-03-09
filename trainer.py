import os

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import get_loader
from utils.utils import get_model
from utils.utils import get_optimizer
from utils.utils import get_loss


def train(model, dataloader, device, optimizer_name, loss_name, lr):
    optimizer_object = get_optimizer(optimizer_name)
    optimizer = optimizer_object(model.parameters(), lr=lr)

    loss_fn = get_loss(loss_name)

    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        bs = len(targets)
        classes = torch.zeros((bs, 10))
        for i in range(bs):
            classes[i][targets[i]] = 1
        classes = classes.to(device)

        outputs = model(inputs)
        loss = loss_fn()(outputs, classes) # LeCun & al. used Maximum Log Likehood

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,preds = torch.max(outputs.data, 1)
        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == targets.data)

    loss = running_loss / 60000
    acc = running_corrects.data.item() / 60000
    print('Training results: Loss: {:.4f} Acc: {:.4f}'.format(
                loss, acc))

    return acc


def eval(model, dataloader, device):
    model.eval()

    running_corrects = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        _,preds = torch.max(outputs.data, 1)
        # statistics
        running_corrects += torch.sum(preds == targets.data)

    acc = running_corrects.data.item() / 10000
  
    return acc


def train_model(args):

    if not args.download and args.data_dir == '../data':
        print("ERROR: please provide the data directory from which to take the data.")
        

    kwargs = {'num_workers': 1, 'pin_memory': True} if (torch.cuda.is_available() and args.use_cuda) else {}
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_cuda) else "cpu")

    loader_class = get_loader(args.dataset)
    loader_object = loader_class(args.data_dir, args.batch_size, args.test_batch_size, 
                                 args.custom_transforms, args.crop_size)

    loader_train = loader_object.get_loader(train=True, download=args.download, kwargs=kwargs)
    loader_eval = loader_object.get_loader(train=False, download=args.download, kwargs=kwargs)

    try:
        if args.save_model and args.save_to is None:
            os.mkdir(args.model)
        elif args.save_to is not None:
            os.mkdir(args.save_to)
    except FileExistsError:
        pass

    model = get_model(args.model)

    learning_curve = np.zeros((3, args.n_epochs))
    learning_curve[0, :] = np.arange(1, args.n_epochs + 1)


    for epoch in range(1, args.n_epochs + 1):
        if args.verbose:
            print("Epoch n°", epoch, ":")
        learning_curve[1, epoch - 1] = train(model, loader_train, device, args.optimizer, args.loss, args.lr)
        learning_curve[2, epoch - 1] = eval(model, loader_eval, device)

        if args.save_model:
            torch.save(model, f"{args.model}/epoch_{epoch}.pt")


    if args.show_plot:
        fig = plt.figure().gca()

        plt.plot(learning_curve[0, :], learning_curve[1, :], label='Training')
        plt.plot(learning_curve[0, :], learning_curve[2, :], label='Evaluation')

        plt.title('Learning curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower left")
        fig.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=10))

        plt.show()
    
    return model
