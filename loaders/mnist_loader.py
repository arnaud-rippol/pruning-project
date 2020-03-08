import os
import numpy as np
import torch

from torchvision import transforms, datasets


class MNISTloader(torch.utils.data.Dataset):

    def __init__(self, data_dir='../data', batch_size=50, test_batch_size=1, custom_transforms=None, list_dir=None,
                 out_name=False, phase=None, crop_size=None):
        
        self.data_dir = data_dir
        self.crop_size = crop_size
        self.out_name = out_name

        self.mean = (0.1307,)
        self.std = (0.3081,)
        self.transforms = self.get_transforms(custom_transforms)

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size


    def get_transforms(self, custom_transforms):
        if custom_transforms:
            return custom_transforms

        else:
            if self.crop_size is None:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ]) 
            else:
                return transforms.Compose([
                    transforms.Resize((self.crop_size, self.crop_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ])

    def get_loader(self, train, download, kwargs):
        if train:
            batch_size = self.batch_size
            shuffle = True
        else:
            batch_size = self.test_batch_size
            shuffle = False

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=train, download=download, transform=self.transforms,
                            batch_size=batch_size, shuffle=shuffle, **kwargs))

        return train_loader