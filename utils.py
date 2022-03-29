import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import numpy as np


def load_dataset(dataset, batch_size, kwargs, dataset_root=None):

    if dataset == 'mnist':
        train_loader = DataLoader(
                    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
                    datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)

    elif dataset == 'fashionmnist':
        train_loader = DataLoader(
                    datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
                    datasets.FashionMNIST('data', train=False, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)

    elif dataset == 'svhn':
        train_loader = DataLoader(
                    datasets.SVHN('data', split='train', download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
                    datasets.SVHN('data', split='test', download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)

    elif dataset == 'celeba':
        ### There is a issue with CelebA on PyTorch code, see https://github.com/pytorch/vision/issues/2262 ###
        transform = transforms.Compose([transforms.CenterCrop(128), transforms.Resize(64), transforms.ToTensor()])
        train_loader = DataLoader(
                    datasets.CelebA(dataset_root, split='train', download=False, transform=transform),
                    batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
                    datasets.CelebA(dataset_root, split='test', download=False, transform=transform),
                    batch_size=batch_size, shuffle=True, **kwargs)

    elif dataset == 'celeba-wb':
        # transform = transforms.Compose([transforms.Resize(64)])
        path_to_npy = os.path.join(dataset_root, "celeba-wb", "CelebA-HQ-128_validation.npy")
        test_images = np.load(path_to_npy, allow_pickle=True) / 255
        test_images = torch.unsqueeze(torch.from_numpy(test_images), dim=1)
        # test_images = transform(test_images)
        test_loader = DataLoader(TensorDataset(test_images.float()), batch_size=batch_size, shuffle=True,
                                 **kwargs)

        return None, test_loader

    return train_loader, test_loader

def compute_mse(x1, x2):

    N = x1.nelement()  # Number of pixels

    return float(torch.sum((x1-x2).pow(2)) / N)


