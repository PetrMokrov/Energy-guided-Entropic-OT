import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from typing import Optional

def make_numpy(X) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if isinstance(X, np.ndarray):
        return X
    return np.asarray(X)

def plot_images(batch, fig_name : Optional[str] = None, name_to_save: Optional[str] = None, dpi:int = 100):
    sh = 1 if fig_name is None else 1.3
    fig, axes = plt.subplots(1, 10, figsize=(10, sh), dpi=dpi)
    if fig_name is not None:
        fig.suptitle(fig_name, fontsize=16)
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch)
    if isinstance(batch, torch.Tensor):
        batch = batch.detach().cpu()
    for i in range(10):
        axes[i].imshow(batch[i].clip(0,1).permute(1,2,0))
        axes[i].set_xticks([]); axes[i].set_yticks([])
    fig.tight_layout(pad=0.1)
    if name_to_save is not None:
        plt.savefig(name_to_save)
    plt.show()

def plot_many_images(multibatch, fig_name : Optional[str]=None, name_to_save : Optional[str]=None, dpi:int = 100):
    sh = 4 if fig_name is None else 4.8
    fig, axes = plt.subplots(4, 10, figsize=(10, sh), dpi=dpi)
    if fig_name is not None:
        fig.suptitle(fig_name, fontsize=16)
    if isinstance(multibatch, np.ndarray):
        multibatch = torch.tensor(multibatch)
    if isinstance(multibatch, torch.Tensor):
        multibatch = multibatch.detach().cpu()
    for i in range(10):
        for j in range(4):
            axes[j, i].imshow(multibatch[i, j].clip(0,1).permute((1,2,0)))
            axes[j, i].set_xticks([]); axes[j, i].set_yticks([])
    fig.tight_layout(pad=0.1)
    if name_to_save is not None:
        plt.savefig(name_to_save)
    plt.show()

def transform_pix_scale(data, inp_pix_range=(-1., 1.), pix_range=(0., 1.)):
    if inp_pix_range is None:
        inp_pix_range = (0., 1.)
    int_data = (data - inp_pix_range[0]) / float(inp_pix_range[1] - inp_pix_range[0])
    int_data *= pix_range[1] - pix_range[0]
    int_data += pix_range[0]
    return int_data

class Dataset2Sampler:

    def __init__(self, dataset, batch_size=64, device='cpu', remove_classes=False):
        self.batch_size = batch_size
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self._iter = iter(self.loader)
        self.device = device
        self.remove_classes = remove_classes

    def _get_batch(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            data = next(self._iter)
        if self.remove_classes:
            return data[0]
        return data
    
    def sample(self, size=None):
        return self(size=size)

    def __call__(self, size=None):
        if size is None:
            return self._get_batch().to(self.device)
        if isinstance(size, tuple):
            size = size[0]
        assert isinstance(size, int)
        assert size > 0
        return torch.cat([
            self._get_batch() for _ in range(int(math.ceil(float(size)/self.batch_size)))])[:size].to(self.device)