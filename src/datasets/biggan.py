import torch
import numpy as np

from pytorch_pretrained_biggan import (
    BigGAN, 
    one_hot_from_names, 
    truncated_noise_sample)

from .utils import transform_pix_scale


def load_biggan(resolution=128, device='cpu'):
    assert resolution in [128, 256, 512]
    model_name = f'biggan-deep-{resolution}'
    return BigGAN.from_pretrained(model_name).to(device)


class BigGANdataGenerator:

    class _LatentSampler:

        def __init__(self, gen):
            self.gen = gen

        def sample(self, size):
            if isinstance(size, tuple):
                assert len(size) == 1
                size = size[0]
            return self.gen.sample_latent(size)
        
        def __call__(self, size):
            return self.sample(size)

    @property
    def latent_sampler(self):
        return self._LatentSampler(self)

    def __init__(
        self, 
        model, 
        cls_idx=None, 
        cls_name=None, 
        batch_size=64, 
        truncation=0.4, 
        device='cpu', 
        pix_range=(0., 1.), 
        sample_latent_code=False
    ) -> None:

        if cls_idx is None:
            assert cls_name is not None
            cls_idx = np.nonzero(one_hot_from_names([cls_name,], batch_size=1)[0])[0].item()
            print('used cls index:', cls_idx)
        assert cls_idx >= 0
        assert cls_idx < 1000
        self.cls_idx = cls_idx
        self.model = model
        self.batch_size=batch_size
        self.truncation=truncation
        self.device = device
        self.pix_range = pix_range
        self.sample_latent_code = sample_latent_code
    
    def latent2data(self, z: torch.Tensor):
        class_vector = torch.zeros([z.size(0), 1000]).to(z)
        class_vector[:, self.cls_idx] = 1.
        return transform_pix_scale(self.model(z, class_vector, self.truncation), pix_range=self.pix_range)
    
    def sample_latent(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch.from_numpy(truncated_noise_sample(truncation=self.truncation, batch_size=batch_size)).to(self.device)
    
    def get_batch(self, batch_size=None):
        z = self.sample_latent(batch_size)
        x = self.latent2data(z)
        if self.sample_latent_code:
            return x, z
        return x
    
    def sample(self, size=None):
        return self(size=size)

    def __call__(self, size=None):
        if isinstance(size, tuple):
            size = size[0]
            assert isinstance(size, int)
        return self.get_batch(size)