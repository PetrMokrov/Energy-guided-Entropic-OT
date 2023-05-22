import torch
import torch.nn as nn
from .models_utils import spectral_norm


class TriangularMLP(nn.Module):

    def __init__(
        self, 
        in_dim,
        out_dim, 
        n_min_neurons=100, 
        n_steps=2, 
        activation_gen = lambda : nn.ReLU(True),
        use_dropout=False):

        super().__init__()
        net_list = []
        curr_dim = in_dim

        for i in range(1, n_steps + 1):
            next_dim = max(n_min_neurons, (2 ** i) * in_dim)
            net_list.append(nn.Linear(curr_dim, next_dim))
            net_list.append(activation_gen())
            if use_dropout:
                net_list.append(nn.Dropout(0.005))
            curr_dim = next_dim

        for i in range(n_steps - 1, 0, -1):
            next_dim = max(n_min_neurons, (2 ** i) * in_dim)
            net_list.append(nn.Linear(curr_dim, next_dim))
            net_list.append(activation_gen())
            if use_dropout:
                net_list.append(nn.Dropout(0.005))
            curr_dim = next_dim

        net_list.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*net_list)

    def forward(self, x):
        return self.net(x)

class WeakMover(TriangularMLP):

    def __init__(
        self, 
        in_dim=4,
        out_dim=2, 
        n_min_neurons=100, 
        n_steps=2, 
        activation_gen = lambda : nn.ReLU(True),
        use_dropout=False):

        super().__init__(
            in_dim, out_dim, n_min_neurons, n_steps, activation_gen, use_dropout)

class Mover(TriangularMLP):

    def __init__(
        self,
        dim=2,
        n_min_neurons=100, 
        n_steps=2, 
        activation_gen = lambda : nn.ReLU(True),
        use_dropout=False):
        super().__init__(
            dim, dim, n_min_neurons, n_steps, activation_gen, use_dropout)


class Discriminator(TriangularMLP):

     def __init__(
        self,
        in_dim=2,
        n_min_neurons=100,
        n_steps=4,
        activation_gen=lambda: nn.ReLU(True)):
        super().__init__(
            in_dim, 1, n_min_neurons, n_steps, activation_gen, False)


class FullyConnectedMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hiddens,
        output_dim,
        activation_gen=lambda: nn.ReLU(),
        sn_iters=0
    ):

        def _SN(module):
            if sn_iters == 0:
                return module
            return spectral_norm(
                module, init=False, zero_bias=False, n_iters=sn_iters)

        assert isinstance(hiddens, list)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hiddens = hiddens

        model = []
        prev_h = input_dim
        for h in hiddens:
            model.append(_SN(nn.Linear(prev_h, h)))
            model.append(activation_gen())
            prev_h = h
        model.append(_SN(nn.Linear(hiddens[-1], output_dim)))
        self.net = nn.Sequential(*model)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x).view(batch_size, self.output_dim)

class FullyConnectedMLPwithConfig(FullyConnectedMLP):

    def __init__(
        self,
        input_dim,
        hiddens,
        output_dim,
        config=None,
        activation_gen=lambda: nn.ReLU(),
        sn_iters=0
    ):
        super().__init__(
            input_dim, hiddens, output_dim, 
            activation_gen=activation_gen, sn_iters=sn_iters)
