import os, sys
sys.path.append("../..")

import gdown
import os
import torch
import numpy as np

from zipfile import ZipFile
from scipy.stats import ortho_group
from torch.utils.data import Dataset, DataLoader
import pathlib

from src.gauss2gauss.analytical_solution import get_D_sigma, get_C_sigma, get_optimal_plan_covariance 
from src.gauss2gauss.distributions import LoaderSampler, RotatedGaussisnLoaderSamplerWithDensity

STATS_DATA_PATH = './statistics'

def get_data_path():
    return os.path.join(pathlib.Path(__file__).parent.resolve(), STATS_DATA_PATH)


def initialize_random_rotated_gaussian(eigenvalues, seed=42):
    np.random.seed(seed)

    dim = len(eigenvalues)
    
    rotation = ortho_group.rvs(dim)
    weight = rotation @ np.diag(eigenvalues)
    sigma = weight @ weight.T

    return weight, sigma


class RotatedGaussianDataset(Dataset):
    def __init__(self, weight, dataset_len, device):
        self.weight = weight.float()
        self.sigma = weight@weight.T
        self.dataset_len = dataset_len
        self.device = device
        self.dim = self.sigma.shape[0]
        
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        return torch.randn(self.dim)@self.weight.T, torch.zeros(self.dim)
        
    @classmethod
    def make_new_dataset(cls, dim=2, dataset_len=int(1e18), device="cpu", eigenvalues=None, seed=42):
        if eigenvalues is None:
            eigenvalues = np.exp(np.linspace(np.log(0.5), np.log(2), dim))
            
        assert len(eigenvalues) == dim
        
        weight, sigma = initialize_random_rotated_gaussian(eigenvalues, seed)
        weight = torch.tensor(weight, device=device)
                
        return cls(weight, dataset_len, device)
    
    @classmethod
    def load_from_tensor(cls, tensor_path, dataset_len=int(1e18), device="cpu"):
        weight = torch.load(tensor_path, map_location=device)
        
        return cls(weight, dataset_len, device)

    def save_to_tensor(self, path):
        torch.save(self.weight, path)
        

def get_rotated_gaussian_dataset(input_or_target, dim, device="cpu"):
    assert input_or_target in ["input", "target"]
    assert dim in [2, 4, 8, 16, 32, 64, 128]
    
    file_name = (f"rotated_gaussian_{dim}_weight_X.torch" 
                 if input_or_target == "input" else f"rotated_gaussian_{dim}_weight_Y.torch"
                )
    
    return RotatedGaussianDataset.load_from_tensor(
        os.path.join(get_data_path(), file_name),
        device=device)


def get_rotated_gaussian_sampler(input_or_target, dim, batch_size, with_density, device="cpu"):
    assert input_or_target in ["input", "target"]
    assert dim in [2, 4, 8, 16, 32, 64, 128]
    
    dataset = get_rotated_gaussian_dataset(input_or_target, dim, device)
    
    if with_density:
        return RotatedGaussisnLoaderSamplerWithDensity(
            DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size), device
        )
    else:
        return LoaderSampler(DataLoader(dataset, shuffle=False, num_workers=8, batch_size=batch_size), device)


def get_rotated_gaussian_benchmark_stats(dim, eps, device="cpu"):
    assert dim in [2, 4, 8, 16, 32, 64, 128]

    X_dataset = RotatedGaussianDataset.load_from_tensor(
        os.path.join(get_data_path(), f"rotated_gaussian_{dim}_weight_X.torch"),
        device=device
    )
    Y_dataset = RotatedGaussianDataset.load_from_tensor(
        os.path.join(get_data_path(), f"rotated_gaussian_{dim}_weight_Y.torch"),
        device=device
    )
    
    covariance_X = X_dataset.sigma.cpu().numpy()
    covariance_Y = Y_dataset.sigma.cpu().numpy()
        
    mu_X = np.zeros(covariance_X.shape[0])
    mu_Y = np.zeros(covariance_X.shape[0])
    
    optimal_plan_mu = np.zeros(covariance_X.shape[0]*2)
    optimal_plan_covariance = get_optimal_plan_covariance(covariance_X, covariance_Y, eps)
    
    return mu_X, mu_Y, covariance_X, covariance_Y, optimal_plan_mu, optimal_plan_covariance
