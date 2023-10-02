import numpy as np
import torch
import torchvision.datasets as TVdatasets
from torchvision import transforms as TVtransforms
import pathlib
import os

file_dir = pathlib.Path(__file__).parent.resolve()

DEFAULT_DATASET_PATH = os.path.join(file_dir, 'data/cmnist')

def random_color(im):
    hue = 360*np.random.rand()
    d = (im *(hue%60)/60)
    im_min, im_inc, im_dec = torch.zeros_like(im), d, im - d
    H = round(hue/60) % 6    
    cmap = [[0, 3, 2], [2, 0, 3], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]]
    return torch.cat((im, im_min, im_dec, im_inc), dim=0)[cmap[H]]


class CMNISTDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        train=True,
        digit='two',
        spat_dim=(28, 28),
        root=DEFAULT_DATASET_PATH,
        download=False,
        pix_range=(-1., 1.)
    ):
        self.digit = digit
        assert digit in ['two', 'three']
        digit_map = {
            'two': 2,
            'three': 3
        }
        _m, _std = pix_range[0]/float(pix_range[0] - pix_range[1]), 1./float(pix_range[1] - pix_range[0])
        TRANSFORM = TVtransforms.Compose([
            TVtransforms.Resize(spat_dim),
            TVtransforms.ToTensor(),
            random_color,
            TVtransforms.Normalize([_m],[_std])
        ])
        mnist = TVdatasets.MNIST(root=root, train=train, download=download, transform=TRANSFORM)
        idx = np.array(range(len(mnist)))
        mnist_digit = torch.utils.data.Subset(mnist, idx[mnist.targets==digit_map[digit]])
        self.mnist_digit = mnist_digit

    def __len__(self):
        return len(self.mnist_digit)

    def __getitem__(self, idx):
        return self.mnist_digit[idx][0]

