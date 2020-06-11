import numpy as np

import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

from sklearn.model_selection import train_test_split

from functools import lru_cache

        
# AVAILABLE_DATASETS = ['mnist', 'cifar10', 'cifar2', 'cifar2_binary', 'stl10', 'imagenet10']
MAX_CACHE_SIZE = 10000


@lru_cache(maxsize=MAX_CACHE_SIZE)
def cached_default_loader(path):
    return datasets.folder.default_loader(path)


class SquareCrop():
    def __call__(self, img):
        return F.center_crop(img, min(img.size))
    
    
class CIFAR2(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(CIFAR2, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.data = self.data[self.targets < 2]
        self.targets = self.targets[self.targets < 2]
        print('dataset size = {}'.format(len(self.data)))
        
        
class ImageNet10(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):

        square_crop = SquareCrop()
        super(ImageNet10, self).__init__(
            root, loader=datasets.folder.default_loader,
            transform=transforms.Compose([square_crop, transform]),
            target_transform=target_transform
        )


def _get_mean_and_std(dataset_name):
    if dataset_name == 'mnist':
        return 0.1307, 0.3081
    elif dataset_name == 'cifar10':
        return 0.4734, 0.2516
    elif dataset_name in ['cifar2', 'cifar2_binary']:
        return 0.4734, 0.2516
    elif dataset_name == 'stl10':
        return np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0
    elif dataset_name.startswith('imagenet10_'):
        return 0.5, 0.5
    else:
        raise ValueError("unknown dataset: {}".format(dataset_name))
        
        
def _get_dataset_class(dataset_name):
    if dataset_name == 'mnist':
        return datasets.MNIST
    elif dataset_name == 'cifar10':
        return datasets.CIFAR10
    elif dataset_name in ['cifar2', 'cifar2_binary']:
        return CIFAR2
    elif dataset_name == 'stl10':
        return datasets.STL10
    elif dataset_name.startswith('imagenet10_'):
        return ImageNet10
    else:
        raise ValueError("unknown dataset: {}".format(dataset_name))
    

def get_shape(dataset_name):
    if dataset_name == 'mnist':
        return (1, 28, 28), 10
    elif dataset_name == 'cifar10':
        return (3, 32, 32), 10
    elif dataset_name in ['cifar2', 'cifar2_binary']:
        return (3, 32, 32), 2
    elif dataset_name == 'stl10':
        return (3, 96, 96), 10
    elif dataset_name.startswith('imagenet10_'):
        return (3, None, None), 10
    else:
        raise ValueError("unknown dataset: {}".format(dataset_name))
        
        
def get_loaders(
    dataset_name, batch_size, train_size=None, shuffle_train=True,
    image_transform=transforms.Compose([]), num_workers=0
):
    dataset_mean, dataset_std = _get_mean_and_std(dataset_name)
    if isinstance(dataset_mean, float):
        dataset_mean = (dataset_mean,)
    if isinstance(dataset_std, float):
        dataset_std = (dataset_std,)
        
    DatasetClass = _get_dataset_class(dataset_name)
    data_path = "data/{}".format(dataset_name.split('_')[0])
    
    transform = transforms.Compose(
        [image_transform, transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std)])
    
    if dataset_name.startswith('imagenet10_'):
        data_path = data_path + '/imagenet_images'
        dataset = DatasetClass(data_path, transform=transform)
        
        if train_size is None:
            raise ValueError("'train_size' should be integer < {} for ImageNet10".format(len(dataset)))
        
        train_indices, test_indices = train_test_split(
            np.arange(len(dataset)), train_size=train_size, stratify=dataset.targets)
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_indices), batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, test_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader_det = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, test_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        if dataset_name == 'stl10':
            train_dataset = DatasetClass(
                data_path, split='train', download=True, transform=transform)
            test_dataset = DatasetClass(
                data_path, split='test', download=True, transform=transform)
        else:
            train_dataset = DatasetClass(
                data_path, train=True, download=True, transform=transform)
            test_dataset = DatasetClass(
                data_path, train=False, download=True, transform=transform)

        if train_size is not None and train_size < len(train_dataset):
            train_indices, _ = train_test_split(
                np.arange(len(train_dataset)), train_size=train_size, stratify=train_dataset.targets)
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader_det = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, test_loader_det