import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random

from LibMTL import Trainer
from LibMTL.loss import CELoss
from LibMTL.metrics import AccMetric
from LibMTL.model import resnet
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from create_dataset import MultiMNISTDataset, MultiMNISTDataset3Digits

def parse_args(parser):
    parser.add_argument('--dataset', default='multimnist', type=str, help='multimnist, multimnist3')
    parser.add_argument('--train_bs', default=64, type=int, help='batch size for training')
    parser.add_argument('--test_bs', default=64, type=int, help='batch size for test')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/', type=str, help='dataset path')
    return parser.parse_args()

def main(params):
    # Set random seed and device
    kwargs, optim_param, scheduler_param = prepare_args(params)
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create MultiMNIST datasets
    if params.dataset == 'multimnist':
        train_multimnist = MultiMNISTDataset(root=params.dataset_path, train=True, transform=transform)
        test_multimnist = MultiMNISTDataset(root=params.dataset_path, train=False, transform=transform)
        task_dict = {'top-left': {'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]},
                    'bottom-right': {'metrics': ['Acc'],
                       'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]}}
    elif params.dataset == 'multimnist3':
        train_multimnist = MultiMNISTDataset3Digits(root=params.dataset_path, train=True, transform=transform)
        test_multimnist = MultiMNISTDataset3Digits(root=params.dataset_path, train=False, transform=transform)
        task_dict = {'top-left': {'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                       'loss_fn': CELoss(),
                       'weight': [1]},
                    'bottom-center': {'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                        'loss_fn': CELoss(),
                        'weight': [1]},
                    'top-right': {'metrics': ['Acc'],
                        'metrics_fn': AccMetric(),
                        'loss_fn': CELoss(),
                        'weight': [1]}}
    else:
        raise ValueError('No support dataset {}'.format(params.dataset))
        
    # Create data loaders
    multimnist_train_loader = DataLoader(train_multimnist, batch_size=params.train_bs, shuffle=True)
    multimnist_test_loader = DataLoader(test_multimnist, batch_size=params.test_bs, shuffle=False)
    
    class MultiLeNetR(nn.Module):
        """The encoder part of the LeNet network adapted to MultiTask Learning. The model consists of two convolutions
        followed by a fully connected layers, resulting in a 50-dimensional embedding."""

        def __init__(self, in_channels=1):
            super().__init__()
            self.in_channels = in_channels
            self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.fc = nn.Linear(320, 50)

        def forward(self, x):
            # x = x.view(-1, 1, 28, 28)
            x = self.conv1(x)
            x = F.relu(F.max_pool2d(x, 2))
            x = self.conv2(x)
            x = F.relu(F.max_pool2d(x, 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc(x))
            return x

        def get_last_layer(self):
            return self.fc


    class MultiLeNetO(nn.Module):
        """The decoder part of the LeNet network adapted to MultiTask Learning. The output has 10 dimensions, since this
        model is used for datasets such as MultiMNIST."""

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(50, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
    
    # Define the decoders
    decoders = nn.ModuleDict({task: MultiLeNetO() for task in list(task_dict.keys())})

    class MultiMNISTtrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, 
                     scheduler_param, **kwargs):
            super(MultiMNISTtrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting, 
                                            architecture=architecture, 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)
    MultiMNISTmodel = MultiMNISTtrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=MultiLeNetR, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          **kwargs)
    # Train the model
    if params.mode == 'train':
        MultiMNISTmodel.train(multimnist_train_loader, multimnist_test_loader, params.epochs)
    elif params.mode == 'test':
        MultiMNISTmodel.test(multimnist_test_loader)
    else:
        raise ValueError

if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)