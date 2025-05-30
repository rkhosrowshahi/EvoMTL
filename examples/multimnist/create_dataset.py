import os
from typing import Callable, Optional
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
import fnmatch
import numpy as np
import random

from tqdm import tqdm

class MultiMNISTDataset(VisionDataset):    
    training_file = "training.pt"
    test_file = "test.pt"
    task_names = ["top-left", "bottom-right"]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    def __init__(
        self,
        root: str,
        samples: int = 60000,
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = samples
        if root.endswith("/"):
            root = root[:-1]
        if samples != 60000:
            root = root + "-" + str(samples)

        print(root)

        self.root = root
        self.train = train
        self._split = "train" if train else "test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not complete." + " You can use download=True to download it")

        # # rgb folder as ground truth
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index].long()

        img = img.view(1, 28, 28)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = tuple(t for t in target)
        # target = {n: t for n, t in zip(self.task_names, target)}
        # return img, target
        return img, {self.task_names[0]: target[0], self.task_names[1]: target[1]}

    def __len__(self):
        return len(self.data)

    def _check_exists(self) -> bool:
        """Only checking for folder existence"""
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.processed_folder, self.test_file)
        )

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download MNIST
        mnist_train = torchvision.datasets.MNIST(
            root=self.root,
            download=True,
            train=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        mnist_test = torchvision.datasets.MNIST(
            root=self.root,
            download=True,
            train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        samples = self.samples
        training_set = create_data_samples(mnist_train, num_samples=samples)
        if samples > 10000:
            samples = 10000
        test_set = create_data_samples(mnist_test, num_samples=samples)

        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")


def create_new_sample(left_idx, right_idx, dataset):
    """Creates new samples for the MultiMNIST dataset. The procesdure is the following: two random MNIST samples with
    dimensions 28x28 are selected and positioned on a 36x36 image, one on the top left corner and the other on the
    bottom right. The pixels in the middle of the generated sample depend on both samples; the maximum value of both is
    selected. The final images results from resizing to 28x28.

    Args:
        left_idx (int): index for left task
        right_idx (int): index for right task
        dataset (Dataset): The dataset to draw samples from.

    Returns:
        tuple(Tensor, int, int): a tuple containing the generated sample as a torch.tensor as well as the associated targets.
    """
    left = dataset[left_idx][0]
    right = dataset[right_idx][0]

    left_label = dataset[left_idx][1]
    right_label = dataset[right_idx][1]

    lim = left.reshape(28, 28)
    rim = right.reshape(28, 28)

    new_im = np.zeros((36, 36))
    new_im[0:28, 0:28] = lim
    new_im[6:34, 6:34] = rim
    new_im[6:28, 6:28] = np.maximum(lim[6:28, 6:28], rim[0:22, 0:22])
    # multi_data_im =  m.imresize(new_im, (28, 28), interp='nearest')
    multi_data_im = np.array(Image.fromarray(new_im).resize(size=(28, 28)))

    multi_data_im = torch.Tensor(multi_data_im)
    return (multi_data_im, left_label, right_label)


def create_data_samples(dataset, num_samples):
    """Generates random samples.

    Args:
        dataset (Dataset): the dataset to generate tasks from.
        num_samples (int): the number of samples tp generate.

    Returns:
        tuple: a tuple containing the samples and the targets.
    """
    dataset_length = num_samples
    # dataset_length = 1000

    print(f"Creating {dataset_length} new samples\n")
    custom_data = torch.zeros(dataset_length, 28, 28)
    labels = torch.zeros(dataset_length, 2)
    for i in tqdm(range(dataset_length)):
        left_idx = np.random.randint(low=0, high=dataset_length)
        right_idx = np.random.randint(low=0, high=dataset_length)
        while left_idx == right_idx:
            left_idx = np.random.randint(low=0, high=dataset_length)
            right_idx = np.random.randint(low=0, high=dataset_length)

        sample = create_new_sample(left_idx, right_idx, dataset)
        custom_data[i, :, :] = sample[0]
        labels[i, :] = torch.LongTensor([sample[1], sample[2]])

    return (custom_data, labels)


class MultiMNISTDataset3Digits(VisionDataset):    
    training_file = "training.pt"
    test_file = "test.pt"
    task_names = ["top-left", "bottom-center", "top-right"]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    def __init__(
        self,
        root: str,
        samples: int = 60000,
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = samples
        if root.endswith("/"):
            root = root[:-1]
        if samples != 60000:
            root = root + "-" + str(samples)

        self.root = root
        self.s = 42
        self.train = train
        self._split = "train" if train else "test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not complete." + " You can use download=True to download it")

        # # rgb folder as ground truth
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index].long()

        img = img.view(1, 28, 28)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = tuple(t for t in target)
        # target = {n: t for n, t in zip(self.task_names, target)}
        # return img, target
        return img, {self.task_names[0]: target[0], self.task_names[1]: target[1], self.task_names[2]: target[2]}

    def __len__(self):
        return len(self.data)

    def _check_exists(self) -> bool:
        """Only checking for folder existence"""
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.processed_folder, self.test_file)
        )

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download MNIST
        mnist_train = torchvision.datasets.MNIST(
            root=self.root,
            download=True,
            train=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        mnist_test = torchvision.datasets.MNIST(
            root=self.root,
            download=True,
            train=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        samples = self.samples
        training_set = create_data_samples(mnist_train, num_samples=samples, orig_img_size=self.s)
        if samples > 10000:
            samples = 10000
        test_set = create_data_samples(mnist_test, num_samples=samples, orig_img_size=self.s)

        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")


def create_new_sample(ids, dataset, s):
    """Creates new samples for the MultiMNIST dataset. The procesdure is the following: two random MNIST samples with
    dimensions 28x28 are selected and positioned on a 36x36 image, one on the top left corner and the other on the
    bottom right. The pixels in the middle of the generated sample depend on both samples; the maximum value of both is
    selected. The final images results from resizing to 28x28.

    Args:
        left_idx (int): index for left task
        right_idx (int): index for right task
        dataset (Dataset): The dataset to draw samples from.

    Returns:
        tuple(Tensor, int, int): a tuple containing the generated sample as a torch.tensor as well as the associated targets.
    """
    x0, y0 = dataset[ids[0]]
    x1, y1 = dataset[ids[1]]
    x2, y2 = dataset[ids[2]]

    img = torch.zeros((s, s))

    img[0:28, 0:28] = x0

    a0, a1 = s - 28, s
    b0, b1 = s // 2 - 14, s // 2 + 14
    img[a0:a1, b0:b1] = torch.max(img[a0:a1, b0:b1], x1)

    a0, a1 = 0, 28
    b0, b1 = s - 28, s
    img[a0:a1, b0:b1] = torch.max(img[a0:a1, b0:b1], x2)

    img = transforms.functional.resize(img.unsqueeze(0), size=28).squeeze()
    return (img, y0, y1, y2)


def create_data_samples(dataset, num_samples, orig_img_size):
    """Generates random samples.

    Args:
        dataset (Dataset): the dataset to generate tasks from.
        num_samples (int): the number of samples tp generate.

    Returns:
        tuple: a tuple containing the samples and the targets.
    """
    dataset_length = num_samples
    # dataset_length = 1000

    print(f"Creating {dataset_length} new samples\n")
    custom_data = torch.zeros(dataset_length, 28, 28)
    labels = torch.zeros(dataset_length, 3)
    for i in tqdm(range(dataset_length)):
        ids = np.random.randint(low=0, high=dataset_length, size=(3,))
        while len(set(ids)) < 3:
            ids = np.random.randint(low=0, high=dataset_length, size=(3,))

        img, y0, y1, y2 = create_new_sample(ids, dataset, s=orig_img_size)
        custom_data[i, :, :] = img
        labels[i, :] = torch.LongTensor([y0, y1, y2])

    return (custom_data, labels)