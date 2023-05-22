import ot
import torch
import torch.distributions as TD
import numpy as np

class DiscreteEOT_l2sq_sampler:

    @staticmethod
    def discrete_sample_conditional(Y, G, i_x, n_pts, return_indices=False):
        probs = G[i_x] / torch.sum(G[i_x])
        distrib = TD.Categorical(probs = probs)
        numbers = distrib.sample((n_pts,))
        if not return_indices:
            return Y[numbers]
        return numbers

    def __init__(self, X, Y, G, device='cpu'):
        self.X = torch.tensor(X).float().to(device)
        self.Y = torch.tensor(Y).float().to(device)
        self.G = torch.tensor(G).float().to(device)
        self.device = device

    def sample(self, x_samples):
        raise NotImplementedError()

    def sample_by_indices(self, x_indices, return_indices=False):
        spls = []
        for x_idx in x_indices:
            spls.append(self.discrete_sample_conditional(self.Y, self.G, x_idx, 1, return_indices=return_indices))
        return torch.cat(spls, dim=0)

    def sample_by_index(self, x_index, n, return_indices=False):
        return self.discrete_sample_conditional(self.Y, self.G, x_index, n, return_indices=return_indices)

def store_discrete_ot(path, model):
    data = {
        'X': model.X.detach().cpu(),
        'Y': model.Y.detach().cpu(),
        'G': model.G.detach().cpu(),
    }
    torch.save(data, path)

def load_discrete_ot(path, device='cpu'):
    CP = torch.load(path)
    return DiscreteEOT_l2sq_sampler(CP['X'], CP['Y'], CP['G'], device=device)

class DiscreteEOT_l2sq:

    def _cast(self, x):
        if self.dtype == 'torch32':
            return torch.tensor(x).float().to(self.device)
        if self.dtype == 'torch64':
            return torch.tensor(x).double().to(self.device)
        if self.dtype == 'np32':
            return make_numpy(x).astype('float32')
        if self.dtype == 'np64':
            return make_numpy(x).astype('float64')

    def __init__(
        self, 
        verbose=False,
        method='sinkhorn_log', 
        stopThr=1e-09,
        numItermax=10000,
        dtype='torch32',
        device='cpu',
    ):
        self.verbose = verbose
        self.method = method
        self.stopThr = stopThr
        self.numItermax = numItermax
        self.dtype = dtype
        self.device = device

    def solve(self, X, Y, eps):
        _X, _Y = self._cast(X), self._cast(Y)
        M = 0.5 * ot.dist(_X, _Y)
        xL, yL = X.shape[0], Y.shape[0]
        wX, wY = self._cast(np.ones(xL)/xL), self._cast(np.ones(yL)/yL)
        G = ot.sinkhorn(
            wX, wY, M, eps, 
            method=self.method, 
            numItermax=self.numItermax, 
            stopThr=self.stopThr, 
            verbose=self.verbose)
        return DiscreteEOT_l2sq_sampler(_X, _Y, G, device=self.device)