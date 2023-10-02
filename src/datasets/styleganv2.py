import torch
import os, sys
from typing import Union, Optional, Tuple, List
import pickle

import warnings
warnings.filterwarnings('ignore')

STYLEGAN_PATH = '../../latentspace/thirdparty/stylegan2_ada_pytorch'
sys.path.append(STYLEGAN_PATH)

from .utils import transform_pix_scale
from ..utils import DataParallelAttrAccess

class StyleGANV2dataGenerator:

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

    dstype2file = {
        'cat': 'afhqcat.pkl',
        'dog': 'afhqdog.pkl',
        'wild': 'afhqwild.pkl',
        'ffhq': 'ffhq.pkl',
        'cmnist3': 'cmnist3.pkl'
    }

    @staticmethod
    def model_to_device(
        model: torch.nn.Module,
        device: str = 'cpu', 
        data_parallel_ids: Optional[List[int]] = None
        ) -> torch.nn.Module:
        if data_parallel_ids is not None:
            return DataParallelAttrAccess(model, device_ids=data_parallel_ids).to('cuda:{}'.format(data_parallel_ids[0]))
            # return DataParallelAttrAccess(model, device_ids=data_parallel_ids).cuda()
        return model.to(device)

    @staticmethod
    def load_model(
        dataset_type: str, 
        model_type: str, 
        device: str ='cpu', 
        data_parallel_ids: Optional[List[int]] = None
        ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        assert dataset_type in ['cat', 'dog', 'wild', 'ffhq', 'cmnist3']
        assert model_type in ['G', 'G_ema']
        PRETRAINED_MODEL_PATH = os.path.join(
            STYLEGAN_PATH, 
            'data/pretrained/', 
            StyleGANV2dataGenerator.dstype2file[dataset_type])

        with open(PRETRAINED_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)[model_type]

        return (
            StyleGANV2dataGenerator.model_to_device(model.mapping, device, data_parallel_ids), 
            StyleGANV2dataGenerator.model_to_device(model.synthesis, device, data_parallel_ids))


    def __init__(
        self,
        dataset_type : str,
        model_type : str,
        map_syn_models : Optional[Tuple[torch.nn.Module, torch.nn.Module]] = None,
        z_dim: int = 512,
        batch_size: int = 32,
        device: str ='cpu', 
        data_parallel_ids: Optional[List[int]] = None,
        pix_range: Tuple[float, float] =(0., 1.), 
        sample_latent_code=False,
        truncation_psi=1.0,  # StyleGAN param
        truncation_cutoff=None, # StyleGAN param
        skip_w_avg_update=True, # StyleGAN param
        noise_mode='const', # StyleGAN param
        force_fp32=True, # StyleGAN param
    ) -> None:
        if map_syn_models is None:
            self.mapping_network, self.synthesis_network = self.load_model(dataset_type, model_type, device, data_parallel_ids)
        else:
            self.mapping_network = self.model_to_device(map_syn_models[0], device, data_parallel_ids)
            self.synthesis_network = self.model_to_device(map_syn_models[1], device, data_parallel_ids)
        self.batch_size = batch_size
        self.pix_range = pix_range
        self.sample_latent_code = sample_latent_code
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.skip_w_avg_update=skip_w_avg_update
        self.noise_mode = noise_mode
        self.force_fp32 = force_fp32
        self.device=device
        self.z_dim = z_dim
    
    def sample_latent(self, batch_size : Optional[int] = None) -> torch.Tensor:
        if batch_size is None:
            batch_size = self.batch_size
        return torch.randn([batch_size, self.z_dim]).to(self.device)
    
    def latent2data(self, z: torch.Tensor):
        c = None
        w = self.mapping_network(
            z, c, 
            truncation_psi=self.truncation_psi, 
            truncation_cutoff=self.truncation_cutoff, 
            skip_w_avg_update=self.skip_w_avg_update)
        imgs = self.synthesis_network(w, noise_mode=self.noise_mode, force_fp32=self.force_fp32)
        return transform_pix_scale(imgs, pix_range=self.pix_range)
    
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


