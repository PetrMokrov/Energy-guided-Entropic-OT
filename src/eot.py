import torch
import torch.nn as nn
import torch.autograd as autograd

import numpy as np
from collections import defaultdict
from typing import Callable, Union, Tuple
import random

from .eot_utils import computePotGrad, evaluating
from .utils import Config, clean_resources

from torchvision import utils as TVutils
import sys
import os

##############
# SampleBuffer

class SampleBufferGeneric:
    
    def __init__(self, noise_gen):
        self.noise_gen = noise_gen
    
    def __len__(self):
        raise NotImplementedError()
    
    def push(self, Xs, samples, ids):
        raise NotImplementedError()
    
    def get(self, n_samples):
        raise NotImplementedError()
    
    def __call__(self, Xs):
        raise NotImplementedError()

class SampleBufferEgEOT(SampleBufferGeneric):

    def __init__(self, noise_gen, p=0.95, max_samples=10000, device='cpu'):
        self.max_samples = max_samples
        self.buffer = []
        self.device = device
        self.p = p
        super().__init__(noise_gen)

    def __len__(self):
        return len(self.buffer)

    def push(self, Xs, samples, ids):
        samples = samples.detach().cpu()
        Xs = Xs.detach().cpu()
        
        if ids is None:
            for sample, X in zip(samples, Xs):
                self.buffer.append((sample, X))

                if len(self.buffer) > self.max_samples:
                    self.buffer.pop(0)
        else:
            assert len(ids) == len(samples)
            assert max(ids) < len(self.buffer)
            samp_Xs = [(sample, X) for sample, X in zip(samples, Xs)]
            for i, _id in enumerate(ids):
                self.buffer[_id] = samp_Xs[i]
            

    def get(self, n_samples):
        indices = random.choices(range(len(self.buffer)), k=n_samples)
        items = [self.buffer[i] for i in indices]
        samples, Xs = zip(*items)
        samples = torch.stack(samples, 0).to(self.device)
        Xs = torch.stack(Xs, 0).to(self.device)
        return Xs, samples, indices

    def get_random(self, Xs):
        samples = self.noise_gen.sample((Xs.size(0),)).to(Xs)
        return Xs, samples, None
    
    def __call__(self, Xs):
        batch_size = Xs.size(0)
        if len(self) < 1:
            return self.get_random(Xs)

        n_replay = (np.random.rand(batch_size) < self.p).sum()

        if n_replay == 0:
            Xs, samples, _ = self.get_random(Xs)
        elif n_replay == batch_size:
            Xs, samples, _ = self.get(n_replay, device=Xs.device)
        else:
            replay_Xs, replay_samples, _ = self.get(n_replay)
            random_Xs, random_samples, _ = self.get_random(Xs[n_replay:])
            Xs, samples = torch.cat([replay_Xs, random_Xs], 0), torch.cat([replay_samples, random_samples], 0)

        return Xs, samples, None


class SampleBufferStatic(SampleBufferGeneric):
    
    def __init__(self, noise_gen, Xs, device='cpu'):
        self.Xs = Xs
        self.noise_gen = noise_gen
        self.Ys = self.noise_gen((Xs.size(0),)).cpu()
        self.device = device
    
    def __len__(self):
        return len(self.Xs)
    
    def push(self, Xs, samples, ids):
        self.Xs[ids] = Xs.detach().cpu()
        self.Ys[ids] = samples.detach().cpu()
        del Xs
        del samples
    
    def get(self, n_samples):
        indices = np.random.choice(len(self), n_samples)
        return self.Xs[indices].to(self.device), self.Ys[indices].to(self.device), indices
    
    def __call__(self, Xs):
        return self.get(len(Xs))

#############################
# Energy sampling techniques

def sample_langevin_batch(
    score_function: Callable,
    x: torch.Tensor, 
    eps=1e-3, # step size (aka ENERGY_SAMPLING_STEP)
    n_steps=100, #        (aka ENERGY_SAMPLING_ITERATIONS)
    decay=1., #           (aka LANGEVIN_DECAY)
    thresh=None, #        (aka LANGEVIN_THRESH)
    noise=0.005, #        (aka LANGEVIN_SAMPLING_NOISE)
    data_projector=lambda x: x.clamp_(0., 1.),
    compute_stats=False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    torch.Tensor, torch.Tensor]:
    '''
    Overall, langevin step looks as:
    X_{t + 1} = X_{t} + 0.5 * eps * score(X_{t}) + noise * N(0, 1)
    '''
    
    # make eps and noise to be data-dimensional
    if isinstance(eps, torch.Tensor):
        assert eps.size(0) == x.size(0)
    elif isinstance(eps, float):
        eps = eps + torch.zeros(x.size(0), device=x.device)

    noise = noise + torch.zeros(x.size(0), device=x.device)

    eps_shape = (x.size(0),) + (1, ) * (x.ndim - 1)
    eps = eps.view(eps_shape)
    noise = noise.view(eps_shape)
    
    # statistics
    r_t = torch.zeros(1).to(x.device)
    cost_r_t = torch.zeros(1).to(x.device)
    score_r_t = torch.zeros(1).to(x.device)
    noise_t = torch.zeros(1).to(x.device)
    
    # langevin iterations

    for s in range(n_steps):

        z_t = torch.randn_like(x)
        sc, cost_part, score_part = score_function(x, ret_stats=True)

        ## adjusting discretization step
        if thresh is None:
            eps_adj = eps
            noise_adj = noise
        else:
            sc_norms = torch.norm(sc.view(x.size(0), -1), dim=-1)
            clip_coeff = torch.ones(x.size(0), device=x.device)
            oul_ids = sc_norms > thresh
            clip_coeff[oul_ids] = thresh / sc_norms[oul_ids]
            eps_adj = eps * clip_coeff.view(eps_shape)
            noise_adj = noise * torch.sqrt(clip_coeff.view(eps_shape))

        ## Langevin dynamics
        x = x + 0.5 * eps_adj * sc +  noise_adj * z_t
        
        # stats calculation
        if compute_stats:
            r_t += (0.5 * eps_adj.view(-1) * sc.data.view(sc.size(0), -1).norm(dim=1)).mean()
            cost_r_t += (0.5 * eps_adj.view(-1) * cost_part.data.view(sc.size(0), -1).norm(dim=1)).mean()
            score_r_t += (0.5 * eps_adj.view(-1) * score_part.data.view(sc.size(0), -1).norm(dim=1)).mean()
            noise_t += (noise.view(-1) * z_t.data.view(z_t.size(0), -1).norm(dim=1)).mean()

        eps *= decay
        noise *= np.sqrt(decay)

        ## Project data to images compact
        x = data_projector(x)

    if not compute_stats:
        return x
    return x, r_t/float(n_steps), cost_r_t/float(n_steps), score_r_t/float(n_steps), noise_t / float(n_steps)

def clip_by_norm(x, norm_thresh):
    assert len(x.shape) > 1
    x_norms = torch.norm(x, dim=tuple(range(x.ndim))[1:], keepdim=True)
    return torch.where(x_norms < norm_thresh, x, x / x_norms * norm_thresh)


def sample_pseudo_langevin_batch(
    score_function : Callable, 
    x : torch.Tensor,
    eps=10.,
    n_steps=60,
    decay=1.0,
    grad_proj_type='value',
    norm_thresh=1.,
    value_thresh=0.01,
    noise=0.005, 
    data_projector=lambda x: x.clamp_(0., 1.),
    compute_stats=False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    torch.Tensor]:

    if isinstance(eps, torch.Tensor):
        assert eps.size(0) == x.size(0)
    elif isinstance(eps, float):
        eps = eps + torch.zeros(x.size(0), device=x.device)

    eps_shape = (x.size(0),) + (1, ) * (x.ndim - 1)
    eps = eps.view(eps_shape)

    noise = torch.empty_like(x).normal_(0, noise)
    
    r_t = torch.zeros(1).to(x.device)
    cost_r_t = torch.zeros(1).to(x.device)
    score_r_t = torch.zeros(1).to(x.device)

    for s in range(n_steps):
        x.data.add_(noise)
        sc, cost_part, score_part = score_function(x, ret_stats=True)

        if grad_proj_type == 'none':
            pass
        elif grad_proj_type == 'value':
            sc.clamp_(-value_thresh, value_thresh)
            cost_part.clamp_(-value_thresh, value_thresh)
        elif grad_proj_type == 'norm':
            sc = clip_by_norm(sc, norm_thresh)
            cost_part = clip_by_norm(cost_part, norm_thresh)
        else:
            raise Exception('unknown proj_type')

        x.data.add_(0.5 * eps * sc.data)
        eps *= decay

        if compute_stats:
            r_t += 0.5 * eps[0].item() * sc.data.view(sc.size(0), -1).norm(dim=1).mean()
            cost_r_t += 0.5 * eps[0].item() * cost_part.data.view(sc.size(0), -1).norm(dim=1).mean()
            score_r_t += 0.5 * eps[0].item() * score_part.data.view(sc.size(0), -1).norm(dim=1).mean()

        ## Project data to images compact
        x = data_projector(x)

    if compute_stats:
        return x, r_t/float(n_steps), cost_r_t/float(n_steps), score_r_t/float(n_steps), noise.view(x.size(0), -1).norm(dim=1).mean()
    return x

##############
## EgEOT implementations

class EgEOT_Generic_Mixin(nn.Module):
    '''
    EgEOT with general cost function generic class
    '''

    @property
    def real_hreg(self):
        '''
        real hreg value
        '''
        return self.config.HREG * self.config.LANGEVIN_SAMPLING_NOISE ** 2 / self.config.LANGEVIN_COST_COEFFICIENT

    def cost_grad_y(self, y, x):
        '''
        returns \nabla_y c(x, y)
        '''
        raise NotImplementedError()

    def cond_score(self, y, x, ret_stats=False):
        with torch.enable_grad():
            y.requires_grad_(True)
            proto_s = self.forward(y)
            s = computePotGrad(y, proto_s)
            assert s.shape == y.shape
        cost_coeff = (1./ self.config.HREG) * (self.config.LANGEVIN_COST_COEFFICIENT / self.config.ENERGY_SAMPLING_STEP)
        cost_part = self.cost_grad_y(y, x) * cost_coeff
        score_part = s * self.config.LANGEVIN_SCORE_COEFFICIENT
        if not ret_stats:
            return score_part - cost_part
        return score_part - cost_part, cost_part, score_part

    def get_samples_energy(
        self, init_y_samples: torch.Tensor, x_samples: torch.Tensor, 
        eps=5e-2, n_steps=10, decay=1.0, compute_stats=False
    ) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
    torch.Tensor, torch.Tensor]:

        def score(y, ret_stats=False):
            return self.cond_score(y, x_samples, ret_stats=ret_stats)

        if not self.config.ENERGY_SAMPLING_NO_PROJECT_DATA:
            data_projector = lambda x: x.clamp_(self.config.PIX_VAL_MIN, self.config.PIX_VAL_MAX)
        else:
            data_projector = lambda x: x

        method = self.config.ENERGY_SAMPLING_METHOD

        if method == 'langevin_classic':
            return sample_langevin_batch(
                score, init_y_samples,
                eps=eps, n_steps=n_steps,
                decay=decay,
                thresh=self.config.LANGEVIN_THRESH,
                noise=self.config.LANGEVIN_SAMPLING_NOISE,
                data_projector=data_projector,
                compute_stats=compute_stats)

        elif method == 'langevin_pseudo':
            return sample_pseudo_langevin_batch(
                score, init_y_samples,
                eps=eps, n_steps=n_steps, decay=decay, 
                grad_proj_type=self.config.PSEUDO_LANGEVIN_GRAD_PROJ_TYPE,
                norm_thresh=self.config.PSEUDO_LANGEVIN_NORM_THRESH,
                value_thresh=self.config.PSEUDO_LANGEVIN_VALUE_THRESH,
                noise=self.config.PSEUDO_LANGEVIN_NOISE,
                data_projector=data_projector, 
                compute_stats=compute_stats)
        else:
            raise Exception('Unknown energy sampling method "{}"'.format(method))

    def __init__(
        self,
        sample_buffer : SampleBufferGeneric,
        config : Config,
        *args, 
        **kwargs
    ):
        self.sample_buffer = sample_buffer
        self.config = config
        super().__init__(*args, config=config, **kwargs)

    def loss(self, xy_samples):
        '''
        x_samples : (bs, *shape)
        y_samples : (bs, *shape)
        '''
        x_samples = xy_samples[0]
        pos_y_samples = xy_samples[1]

        # slightly noise the data
        if self.config.REFERENCE_DATA_NOISE_SIGMA > 0.0:
            pos_y_samples += self.config.REFERENCE_DATA_NOISE_SIGMA * torch.randn_like(pos_y_samples)
        
        x_samples, neg_y_samples_0, indices = self.sample_buffer(x_samples)

        with evaluating(self):
            with torch.no_grad():
                neg_y_samples, r_t, cost_r_t, score_r_t, noise_norm = self.get_samples_energy(
                    neg_y_samples_0, x_samples, 
                    self.config.ENERGY_SAMPLING_STEP, 
                    self.config.ENERGY_SAMPLING_ITERATIONS,
                    decay=self.config.LANGEVIN_DECAY, compute_stats=True)
        
        self.sample_buffer.push(x_samples, neg_y_samples, indices)
        pos_out = self.forward(pos_y_samples)
        pos_out_mean = pos_out.mean()
        neg_out = self.forward(neg_y_samples)
        neg_out_mean = neg_out.mean()
        loss = - pos_out_mean + neg_out_mean
        loss += self.config.ALPHA * (pos_out.pow(2) + neg_out.pow(2)).mean()
        self.sample_buffer.push(x_samples, neg_y_samples, indices)
        return {
            'pos_out': pos_out_mean,
            'neg_out': neg_out_mean,
            'loss': loss,
            'r_t': r_t,
            'cost_r_t': cost_r_t,
            'score_r_t': score_r_t,
            'noise': noise_norm
        }

    def sample(
        self, x_samples: torch.Tensor, n_iterations=None, step_size=None, decay=None,
        y_init=None, init_sigma=1., init_sampler=None
    ) -> torch.Tensor:
        n_iterations = self.config.ENERGY_SAMPLING_ITERATIONS if n_iterations is None else n_iterations
        step_size = self.config.ENERGY_SAMPLING_STEP if step_size is None else step_size
        decay = self.config.LANGEVIN_DECAY if decay is None else decay

        with evaluating(self):
            with torch.no_grad():
                # sample from initial distribution
                if y_init is not None:
                    z = y_init
                else:
                    if init_sampler is None:
                        z = torch.randn_like(x_samples) * init_sigma
                    else:
                        z = init_sampler.sample(x_samples.size(0)).to(x_samples)
                z = self.get_samples_energy(
                    z, x_samples, eps=step_size, 
                    n_steps=n_iterations, decay=decay)
                assert isinstance(z, torch.Tensor)
                return z

    def store(self, path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        torch.save(
            {
                'state_dict': self.state_dict(),
                'config_dict': self.config.__dict__
            }, path
        )

    @staticmethod
    def load():
        raise NotImplementedError()


class EgEOT_l2sq_Mixin(EgEOT_Generic_Mixin):
    '''
    EgEOT for squared l2 loss $0.5 \\Vert x - y \\Vert_2^2$
    '''

    def cost_grad_y(self, y, x):
        return y - x


class EgEOT_no_cost_Mixin(EgEOT_Generic_Mixin):
    '''
    EgEOT for zero cost (i.e. recovers simple energy based model)
    '''

    def cost_grad_y(self, y, x):
        return 0.

class EgEOT_l2sq_ambient_Mixin(EgEOT_Generic_Mixin):
    '''
    '''
    def __init__(
        self,
        latent2data_gen,
        sample_buffer,
        config,
        *args, 
        **kwargs
    ):
        self.latent2data_gen = latent2data_gen
        super().__init__(sample_buffer, config, *args, **kwargs)

    def cost_grad_y(self, y, x):
        with torch.enable_grad():
            y.requires_grad_(True)
            cost = 0.5 * torch.flatten(self.latent2data_gen(y) - x, start_dim=1).pow(2).sum(dim=1, keepdim=True)
            assert cost.shape == torch.Size([y.size(0), 1])
            res = computePotGrad(y, cost)
        return res
