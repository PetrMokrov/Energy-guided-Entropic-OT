import argparse
import numpy as np
import wandb
from copy import deepcopy

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as TD
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms as TVtransforms

import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append('..')

# gauss2gauss
from src.gauss2gauss.dataset import (
    get_rotated_gaussian_dataset,
    get_rotated_gaussian_sampler,
    get_rotated_gaussian_benchmark_stats
)
from src.gauss2gauss.metrics import compute_BW_UVP_with_gt_stats
from src.gauss2gauss.utils import TrainingScheduler_BW_UVP_Mixin

# dgm_utils
from dgm_utils import train_model
from dgm_utils import StatsManager, StatsManagerDrawScheduler
from dgm_utils.scheduler import (
    TrainingSchedulerSM_Mixin, 
    TrainingSchedulerGeneric, 
    TrainingSchedulerWandB_Mixin
)

# pit utils and other utils
from src.utils import make_numpy
from src.utils import Distrib2Sampler, JointSampler
from src.utils import DataLoaderWrapper
from src.models import FullyConnectedMLP, FullyConnectedMLPwithConfig
from src.utils import Config
from src.utils import ParametersSpecificator

# EOT
from src.eot import EgEOT_l2sq_Mixin
from src.eot import SampleBufferEgEOT
from src.eot_utils import conditional_sample_from_EgEOT

#random seed
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

def reset_seed(seed = 0xBADBEEF):
    OUTPUT_SEED = seed
    torch.manual_seed(OUTPUT_SEED)
    np.random.seed(OUTPUT_SEED)

WANDB_PROJECT_NAME = 'eot'


##########################
## Parsing arguments

CONFIG = Config()


parser = argparse.ArgumentParser(
    description='Gaussian2gaussian benchmark for EgEOT',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# genereal settings

# REQUIRED ARGUMENTS
parser.add_argument('experiment', help='experiment name')
parser.add_argument('--eps', type=float, help='entropy regularization coefficient')
parser.add_argument('--dim', type=int, help='problem dimensionality')

CONFIG.DIM = 2
CONFIG.EPS = 1.0

parser.add_argument('--verbose', dest='verbose', action='store_const', const=True, default=False)
parser.add_argument('--use_wandb', action='store_const', const=True, default=False)

CONFIG.USE_WANDB = True

# training parameters
parser.add_argument('--batch_size', type=int, help='batch size', default=1024)
parser.add_argument('--epochs', type=int, help='number of epochs', default=200)

CONFIG.BATCH_SIZE = 1024
CONFIG.EPOCHS = 200

# link to .yaml config
# the properties given in the config are of the top priority
parser.add_argument('--config', type=str, help='path to the configs (if provided)', action='append', nargs='+', default=None)

# device
parser.add_argument('--device', action='store', help='device (for NN training)', type=str, default='cuda:0')


args = parser.parse_args()

args_dict = deepcopy(args.__dict__)

# DEVICE SETTING
if args_dict['device'].startswith('cuda'):
    DEVICE = 'cuda'
    GPU_DEVICE = int(args_dict['device'].split(':')[1])
    torch.cuda.set_device(GPU_DEVICE)
else:
    DEVICE = 'cpu'

del args_dict['device']

# set up the config with parser.args provided
_n_set = 0
for arg in list(args_dict.keys()):
    if arg.upper() in CONFIG.__dict__.keys():
        val = args_dict[arg]
        if isinstance(val, list):
            val = tuple(val)
        setattr(CONFIG, arg.upper(), val)
        _n_set += 1
assert _n_set == len(list(CONFIG.__dict__.keys()))

# OTHER DEFAULT CONFIG PARAMETERS

CONFIG.CLIP_GRADS_NORM = False
CONFIG.ALPHA = 0.000 # loss += alpha * (pos_out ** 2 + neg_out ** 2)
CONFIG.HREG = CONFIG.EPS

CONFIG.ENERGY_SAMPLING_METHOD = 'langevin_classic'

CONFIG.LANGEVIN_THRESH = None
langevin_sampling_noise_PS = ParametersSpecificator(0.1, {(128, 0.1): 0.1})
CONFIG.LANGEVIN_SAMPLING_NOISE = langevin_sampling_noise_PS(CONFIG.DIM, CONFIG.EPS)
energy_sampling_iterations_PS = ParametersSpecificator(
    100, {(128, 0.1): 100})
CONFIG.ENERGY_SAMPLING_ITERATIONS = energy_sampling_iterations_PS(CONFIG.DIM, CONFIG.EPS)
test_energy_sampling_iterations_PS = ParametersSpecificator(
    700,
    {
        (128, 10.): 700,
        (128, 1.): 700,
        (128, 0.1): 700
    }
)
CONFIG.TEST_ENERGY_SAMPLING_ITERATIONS = test_energy_sampling_iterations_PS(CONFIG.DIM, CONFIG.EPS)
CONFIG.LANGEVIN_DECAY = 1.
CONFIG.LANGEVIN_SCORE_COEFFICIENT = 1.0
CONFIG.LANGEVIN_COST_COEFFICIENT = langevin_sampling_noise_PS(CONFIG.DIM, CONFIG.EPS) ** 2
CONFIG.ENERGY_SAMPLING_STEP = 1.0
CONFIG.SPECTRAL_NORM_ITERS = 0
CONFIG.REFERENCE_DATA_NOISE_SIGMA = 0.00
CONFIG.P_SAMPLE_BUFFER_REPLAY = 0.95
CONFIG.ENERGY_SAMPLING_NO_PROJECT_DATA = True

CONFIG.PSEUDO_LANGEVIN_GRAD_PROJ_TYPE = 'none' # 'value', 'norm', 'none'
CONFIG.PSEUDO_LANGEVIN_NORM_THRESH = 1.
CONFIG.PSEUDO_LANGEVIN_VALUE_THRESH = 0.01
CONFIG.PSEUDO_LANGEVIN_NOISE = 0.005

# model parameters
CONFIG.NN_N_LAYERS = 3
CONFIG.NN_N_NEURONS = 512
CONFIG.ACTIVATION = 'ReLU'


# learning parameters
lr_PS = ParametersSpecificator(
    2e-5, # default parameter
    {
        (128, 10.): 5e-5,
        (128, 1.): 5e-5,
        (128, 0.1): 2e-4,
        (64, 10.): 2e-5,
        (64, 1.): 4e-5,
        (64, 0.1): 7e-5,
        (16, 1.): 1e-5,
        (16, 10.): 4e-6,
        (16, 0.1): 2e-5,
        (2, 1.): 4e-7,
        (2, 10.): 2e-7,
        (2, 0.1): 5e-7
    }
)
CONFIG.LR = lr_PS(CONFIG.DIM, CONFIG.EPS)
CONFIG.ADAM_BETAS = (0.8, 0.99)
USE_CUDA = DEVICE == 'cuda'
CONFIG.SAMPLE_BUFFER_SIZE = 10000
CONFIG.BASIC_NOISE_VAR = 1.0
CONFIG.GT_X_N_SAMPLES = 100000 # number of reference samples to estimate 


# function to preprocess wandb-style configs:
def preprocess_config_from_file(config):
    if '_wandb' in config.keys():
        assert 'wandb_version' in config.keys()
        del config['_wandb']
        del config['wandb_version']
        for key in config.keys():
            config[key] = config[key]['value']
    return config


# updating the config with the arguments in the .yaml convig (if provided)
if args.config is not None:
    for _config in args.config[0]:
        with open(_config, 'r') as fp:
            config_from_file = yaml.full_load(fp)
        config_from_file = preprocess_config_from_file(config_from_file)
        for arg, val in config_from_file.items():
            if arg.upper() in CONFIG.__dict__.keys():
                if isinstance(val, list):
                    val = tuple(val)
                setattr(CONFIG, arg.upper(), val)
            else:
                raise Exception(
                    'CONFIG has not "{}" argument'.format(arg.upper()))

###################################
## Load datasets

X_sampler = get_rotated_gaussian_sampler(
    "input", CONFIG.DIM, with_density=False, batch_size=CONFIG.BATCH_SIZE)
Y_sampler = get_rotated_gaussian_sampler(
    "target", CONFIG.DIM, with_density=False, batch_size=CONFIG.BATCH_SIZE)

reset_seed()
gt_X_sample = torch.cat([
    X_sampler.sample(CONFIG.BATCH_SIZE).detach() for _ in range(
        int(np.ceil(CONFIG.GT_X_N_SAMPLES/float(CONFIG.BATCH_SIZE))))])[:CONFIG.GT_X_N_SAMPLES]

######################
## EOT_IGEBM model

NN_HIDDENS = [CONFIG.NN_N_NEURONS,] * CONFIG.NN_N_LAYERS 
if CONFIG.ACTIVATION == 'ReLU':
    ACTIVATION = lambda: nn.ReLU(True)
else:
    raise Exception('Unsupported activation "{}"'.format(CONFIG.ACTIVATION))

class EgEOT_l2sq(EgEOT_l2sq_Mixin, FullyConnectedMLPwithConfig):

    def __init__(self, sample_buffer, config, 
        dim=CONFIG.DIM,
        hiddens=NN_HIDDENS,
        activation=ACTIVATION):
        super().__init__(
            sample_buffer, 
            config, 
            dim, 
            hiddens, 
            1, 
            activation_gen=activation
        ) # MLP parameters


#######################
## Training scheduler

class TrainingSchedulerGauss2Gauss(
    TrainingScheduler_BW_UVP_Mixin,
    TrainingSchedulerWandB_Mixin
):

    def __init__(
        self,
        model,
        config,
        use_wandb=CONFIG.USE_WANDB,
        estimate_bw_uvp_X_samples=gt_X_sample,
        bw_uvp_stats=get_rotated_gaussian_benchmark_stats(CONFIG.DIM, CONFIG.EPS)
    ):
        self.model = model
        self.config = config
        super().__init__(
            use_wandb = use_wandb,
            estimate_bw_uvp_X_samples=estimate_bw_uvp_X_samples,
            save_bw_uvp_interval=1,
            bw_uvp_apply_mode='test',
            bw_uvp_stats=bw_uvp_stats
        )

###################
## Training

basic_noise_gen = TD.Normal(
    torch.zeros(CONFIG.DIM).to(DEVICE), 
    torch.ones(CONFIG.DIM).to(DEVICE) * CONFIG.BASIC_NOISE_VAR)

sample_buffer = SampleBufferEgEOT(
    basic_noise_gen,
    p=CONFIG.P_SAMPLE_BUFFER_REPLAY,
    max_samples=CONFIG.SAMPLE_BUFFER_SIZE,
    device=DEVICE)

reset_seed(seed=0xCAFFEE)

model = EgEOT_l2sq(sample_buffer, CONFIG)
model.potential = model.potential.to(DEVICE)

train_loader = DataLoaderWrapper(
    JointSampler(X_sampler, Y_sampler),
    CONFIG.BATCH_SIZE, n_batches=100)

test_loader = DataLoaderWrapper(
    JointSampler(X_sampler, Y_sampler),
    CONFIG.BATCH_SIZE, n_batches=10, store_dataset=True)

if CONFIG.USE_WANDB:
    wandb.init(name=args_dict['experiment'], project=WANDB_PROJECT_NAME, reinit=True, config=CONFIG)

scheduler = TrainingSchedulerGauss2Gauss(model, CONFIG)

reset_seed(seed=0xBABDEE)

train_model(
    model, 
    train_loader, 
    test_loader, 
    epochs=CONFIG.EPOCHS, 
    lr=CONFIG.LR,
    adam_betas=CONFIG.ADAM_BETAS,
    use_tqdm=args.verbose, 
    use_cuda=USE_CUDA,
    loss_key='loss',
    conditional=True,
    scheduler=scheduler
)

if CONFIG.USE_WANDB:
    wandbdir = wandb.run.dir
    wandbid = wandb.run.id

    CONFIG.WANDB_ID = wandbid

    wandb.finish()

    # model.store(os.path.join(wandbdir, 'model/model.pth'))
