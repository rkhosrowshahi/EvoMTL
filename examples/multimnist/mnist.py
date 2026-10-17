import os
import pickle
import warnings
from typing import Any, Callable, Optional
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


def _load_multimnist_pickle(path: str) -> Any:
    """Load legacy multi_mnist pickles without NumPy 2.4+ dtype(align=...) deprecation noise."""
    try:
        from numpy.exceptions import VisibleDeprecationWarning
    except ImportError:
        VisibleDeprecationWarning = getattr(  # type: ignore[misc]
            np, "VisibleDeprecationWarning", UserWarning
        )
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", VisibleDeprecationWarning)
            return pickle.load(f)


# Pareto MTL pickles (36x36) per dataset, and the source of the top-left / bottom-right item of the generated
# 28x28 versions (MNIST digit top-left and Fashion item bottom-right for fashion_and_mnist, as in the pickle).
PICKLES = {"mnist": "multi_mnist.pickle", "fashion": "multi_fashion.pickle",
           "fashion_and_mnist": "multi_fashion_and_mnist.pickle"}
SOURCES = {"mnist": ("mnist", "mnist"), "fashion": ("fashion", "fashion"), "fashion_and_mnist": ("mnist", "fashion")}


def _imresize_nearest(arr: np.ndarray, size) -> np.ndarray:
    """``scipy.misc.imresize(arr, size, interp='nearest')`` (SciPy < 1.3) for a 2-D float array: byte-scale to
    0-255 as ``scipy.misc.bytescale`` did, then resize with PIL's NEAREST filter."""
    cmin, cmax = float(arr.min()), float(arr.max())
    scale = 255.0 / (cmax - cmin) if cmax > cmin else 255.0
    byte = (((arr - cmin) * scale).clip(0, 255) + 0.5).astype(np.uint8)
    return np.asarray(Image.fromarray(byte).resize((size[1], size[0]), resample=Image.NEAREST))


def sener_koltun_images(top: np.ndarray, bottom: np.ndarray, rng: np.random.RandomState):
    """Two-item images as in Sener & Koltun's ``read_image_file`` (multi_task/loaders/multi_mnist_loader.py).

    Every image of ``top`` (in order) gets one random partner from ``bottom``; on a 36x36 canvas the first sits at
    [0:28, 0:28] and the partner at [6:34, 6:34], the overlap keeps the brighter pixel, and the canvas is resized to
    28x28 by nearest-neighbour sampling. Returns uint8 images ``[N, 28, 28]`` and the partner indices.
    """
    n = len(top)
    images = np.zeros((n, 28, 28), dtype=np.uint8)
    partners = np.zeros(n, dtype=np.int64)
    for left in tqdm(range(n), desc="Sener & Koltun MultiMNIST", leave=False):
        right = rng.permutation(len(bottom))[:1][0]  # as in the original: np.random.permutation(length)[:1]
        lim, rim = top[left], bottom[right]
        new_im = np.zeros((36, 36))
        new_im[0:28, 0:28] = lim
        new_im[6:34, 6:34] = rim
        new_im[6:28, 6:28] = np.maximum(lim[6:28, 6:28], rim[0:22, 0:22])
        images[left] = _imresize_nearest(new_im, (28, 28))
        partners[left] = right
    return images, partners


def _raw_items(root: str, source: str, train: bool):
    """MNIST or FashionMNIST images (uint8 ``[N, 28, 28]``) and labels, downloaded under ``root`` if missing."""
    cls = torchvision.datasets.MNIST if source == "mnist" else torchvision.datasets.FashionMNIST
    ds = cls(root=root, train=train, download=True)
    return ds.data.numpy(), ds.targets.numpy()


class MultiMNISTDataset(VisionDataset):
    """Two-item MultiMNIST with tasks ``top-left`` and ``bottom-right``.

    ``dataset`` selects the version:

    * ``mnist``, ``fashion``, ``fashion_and_mnist`` (default): 28x28 images made with Sener & Koltun's procedure
      (:func:`sener_koltun_images`) on first use from MNIST / FashionMNIST, 60,000 training and 10,000 test images,
      stored under ``root/data/MultiMNISTDataset/<dataset>/``. Items go through ``transform`` as PIL images, as in their
      loader (ToTensor and Normalize((0.1307,), (0.3081,)) in ``main.py``).
    * the same names with ``-36p``: the 36x36 Pareto MTL pickles under ``root/data/`` (120,000 training and 20,000
      test images), returned as raw 0-255 tensors without ``transform``, as in the Pareto MTL code.
    """

    task_names = ["top-left", "bottom-right"]

    def __init__(
        self,
        root: str,
        dataset: str = "mnist",
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 0,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.dataset, self.train = dataset, train
        base = dataset[: -len("-36p")] if dataset.endswith("-36p") else dataset
        if base not in SOURCES:
            raise ValueError(f"Unknown MultiMNIST dataset {dataset!r}; options: {sorted(SOURCES)}, "
                             "each also with -36p for the 36x36 pickles")
        self.pixels = 36 if dataset.endswith("-36p") else 28
        if self.pixels == 36:
            trainX, trainLabel, testX, testLabel = _load_multimnist_pickle(os.path.join(root, "data", PICKLES[base]))
            X, Y = (trainX, trainLabel) if train else (testX, testLabel)
            self.data = torch.from_numpy(X.reshape(-1, 1, 36, 36)).float()
            self.targets = torch.from_numpy(Y).long()
        else:
            folder = os.path.join(root, "data", self.__class__.__name__, base)
            path = os.path.join(folder, "training.pt" if train else "test.pt")
            if not os.path.exists(path):
                self._generate(root, base, folder, seed)
            self.data, self.targets = torch.load(path)

    @staticmethod
    def _generate(root: str, base: str, folder: str, seed: int) -> None:
        """Make both splits once (training then test) with a seeded generator."""
        rng = np.random.RandomState(seed)
        os.makedirs(folder, exist_ok=True)
        for train, name in ((True, "training.pt"), (False, "test.pt")):
            (top, top_y), (bottom, bottom_y) = (_raw_items(os.path.join(root, "data"), s, train) for s in SOURCES[base])
            images, partners = sener_koltun_images(top, bottom, rng)
            targets = torch.stack([torch.from_numpy(top_y), torch.from_numpy(bottom_y[partners])], dim=1).long()
            torch.save((torch.from_numpy(images), targets), os.path.join(folder, name))

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index].long()
        if self.pixels == 36:  # Pareto MTL pickles: raw 0-255 tensor, no transform
            img = img.view(1, 36, 36)
        else:  # Sener & Koltun: PIL image through the transform
            img = Image.fromarray(img.numpy())
            img = self.transform(img) if self.transform is not None else torch.from_numpy(
                np.asarray(img, dtype=np.float32))[None]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, {self.task_names[0]: target[0], self.task_names[1]: target[1]}

    def __len__(self):
        return len(self.data)


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
        dataset: str = 'mnist',
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

        self.root = os.path.join(root, "data")  # MNIST download and generated files under root/data/
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

        else:
            raise NotImplementedError("Download is not supported for MultiMNIST3Digits")

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
        training_set = create_data_samples_3digits(mnist_train, num_samples=samples, orig_img_size=self.s)
        if samples > 10000:
            samples = 10000
        test_set = create_data_samples_3digits(mnist_test, num_samples=samples, orig_img_size=self.s)

        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")


def create_new_sample_3digits(ids, dataset, s):
    """Creates new samples for the MultiMNIST dataset. The procedure is the following: two random MNIST samples with
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


def create_data_samples_3digits(dataset, num_samples, orig_img_size):
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

        img, y0, y1, y2 = create_new_sample_3digits(ids, dataset, s=orig_img_size)
        custom_data[i, :, :] = img
        labels[i, :] = torch.LongTensor([y0, y1, y2])

    return (custom_data, labels)