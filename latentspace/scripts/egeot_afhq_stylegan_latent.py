import torch
import torch.nn as nn

import pathlib
import argparse
import os
import sys
import string
import random
import wandb

import warnings
warnings.filterwarnings("ignore")

sys.path.append('../..')

import warnings
warnings.filterwarnings('ignore')

# dgm_utils
from dgm_utils import train_model
from dgm_utils import StatsManager, StatsManagerDrawScheduler
from dgm_utils.scheduler import (
    TrainingSchedulerSM_Mixin, 
    TrainingSchedulerModelsSaver_Mixin,
    TrainingSchedulerWandB_Mixin,
    TrainingSchedulerLR_Mixin
)

# other utils
from src.utils import clean_resources
from src.utils import JointSampler
from src.utils import DataLoaderWrapper
from src.utils import Config
from src.utils import get_changing_values_range
from src.models import FullyConnectedMLPwithConfig, ResNet_Dambient_withConfig
from src.utils import DataParallelAttrAccess, clean_resources

# datasets
from src.datasets.utils import transform_pix_scale
from src.datasets.styleganv2 import StyleGANV2dataGenerator
from src.datasets.afhq import get_afhq_dataset
from src.datasets.utils import Dataset2Sampler

# EOT
from src.eot import EgEOT_l2sq_ambient_Mixin
from src.eot import SampleBufferGeneric, SampleBufferStatic
from src.eot_utils import TrainingSchedulerWandB_EgEOT_Mixin

THIS_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# specify wandb project name, if you plan to use wandb
WANDB_PROJECT_NAME = 'eot'

CONFIG = Config()

parser = argparse.ArgumentParser(
    description='EgEOT AFHQ Source -> AFHQ Target in the latent space of StyleGANv2-ada',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# experiment name
parser.add_argument('experiment', help='experiment name')
# source data
parser.add_argument('--source', type=str, default='cat')
# target data
parser.add_argument('--target', type=str, default='dog')
# entropic regularization coefficient
parser.add_argument('--hreg', type=float, help='entropy regularization coefficient', default=1.0)
# f potential parameterization
parser.add_argument('--ambient', action='store_const', const=True, default=False)
# wheather to use wandb or not
parser.add_argument('--use_wandb', action='store_const', const=True, default=False)
# the models (checkpoints) will be stored to $THIS_SCRIPT_DIR\models\$models_dir if models_dir 
# is specified else to $THIS_SCRIPT_DIR\models\[random_name]
parser.add_argument('--models_dir', type=str, default=None)

#########################
### training parameters

# batch_size=128 is ok for 4 A100. Adjust it correspondingly for your hardware setup
parser.add_argument('--batch_size', type=int, help='batch size', default=128)
# for obtaining the images in the rebuttal we train the model for 11 epochs.
parser.add_argument('--epochs', type=int, help='number of epochs', default=11)

# how often to save losses to wandb
parser.add_argument('--save_metrics_interval', type=int, help='frequency of test metrics saving', default=1)

###################
### device. specify several gpu_ids in order to train model on several gpus
parser.add_argument('--gpu_ids', type=int, nargs='*', default=[0, 1, 2, 3])

# some critical parameters we account for
parser.add_argument('--langevin_sampling_noise', type=float, default=0.008)
parser.add_argument('--energy_sampling_iterations', type=int, default=100)
parser.add_argument('--langevin_score_coefficient', type=float, default=0.1)
parser.add_argument('--adam_betas', nargs=2, type=float, default=(0.2, 0.999))
parser.add_argument('--alpha', type=float, default=0.000)

# learning rate parameters
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_fin', type=float, default=0.00001)
parser.add_argument('--lr_coldstart', type=int, default=5)
parser.add_argument('--lr_freeze_final', type=int, default=10)

######################
# parsing arguments

args_dict = parser.parse_args().__dict__

# DEVICE SETTING 
GPU_IDS = args_dict['gpu_ids']
MULTI_GPU_FLAG = len(GPU_IDS) > 1
if len(GPU_IDS) == 1:
    DEVICE = 'cuda'
    GPU_DEVICE = GPU_IDS[0]
    torch.cuda.set_device(GPU_DEVICE)
else:
    DEVICE = 'cuda:{}'.format(GPU_IDS[0])

del args_dict['gpu_ids']

# CONFIG

CONFIG = Config()

# problem setup parameters
CONFIG.SOURCE = 'cat'
CONFIG.TARGET = 'dog'
CONFIG.HREG = 1.0

# technical parameters
CONFIG.MODELS_DIR = None
CONFIG.AMBIENT = False
CONFIG.USE_WANDB = True
CONFIG.CLIP_GRADS_NORM = False
CONFIG.ALPHA = 0.00 # loss += alpha * (pos_out ** 2 + neg_out ** 2)
CONFIG.SPECTRAL_NORM_ITERS = 0

# langevin dynamics parameters
CONFIG.ENERGY_SAMPLING_METHOD = 'langevin_classic' #'classic', 'pseudo'
CONFIG.LANGEVIN_THRESH = None
CONFIG.LANGEVIN_SAMPLING_NOISE = 0.008
CONFIG.ENERGY_SAMPLING_ITERATIONS = 100
CONFIG.TEST_ENERGY_SAMPLING_ITERATIONS = CONFIG.ENERGY_SAMPLING_ITERATIONS * 10
CONFIG.LANGEVIN_DECAY = 1.0
CONFIG.LANGEVIN_SCORE_COEFFICIENT = 0.1
CONFIG.LANGEVIN_COST_COEFFICIENT = CONFIG.LANGEVIN_SAMPLING_NOISE ** 2
CONFIG.ENERGY_SAMPLING_STEP = 1.0
CONFIG.REFERENCE_DATA_NOISE_SIGMA = 0.00
CONFIG.ENERGY_SAMPLING_NO_PROJECT_DATA = True

#-----------------------------------------------
# nonactual for langevin_classic sampling method
CONFIG.PSEUDO_LANGEVIN_GRAD_PROJ_TYPE = 'none' # 'value', 'norm', 'none'
CONFIG.PSEUDO_LANGEVIN_NORM_THRESH = 1.
CONFIG.PSEUDO_LANGEVIN_VALUE_THRESH = 0.01
CONFIG.PSEUDO_LANGEVIN_NOISE = 0.005
#---------------------------------------------

# replay buffer parameters
CONFIG.SAMPLE_BUFFER_TYPE = 'static'
CONFIG.P_SAMPLE_BUFFER_REPLAY = 0.95 # nonactual

# learning parameters
CONFIG.EPOCHS = 100 
CONFIG.BATCH_SIZE = 128 #TODO
CONFIG.ADAM_BETAS = (0.2, 0.99)
CONFIG.SAMPLE_BUFFER_SAMPLES = 10000 # nonactual
CONFIG.BASIC_NOISE_VAR = 1.0
CONFIG.PIX_VAL_MIN = 0.
CONFIG.PIX_VAL_MAX = 1.

CONFIG.LR = 1e-4
CONFIG.LR_FIN = 1e-5
CONFIG.LR_COLDSTART = 5
CONFIG.LR_FREEZE_FINAL = 10

# set up the config with parser.args provided
CONFIG.set_attributes(args_dict)
CONFIG.LANGEVIN_COST_COEFFICIENT = CONFIG.LANGEVIN_SAMPLING_NOISE ** 2
CONFIG.TEST_ENERGY_SAMPLING_ITERATIONS = min(CONFIG.ENERGY_SAMPLING_ITERATIONS * 10, 2000)
if CONFIG.LR_FIN is None:
    CONFIG.LR_FIN = CONFIG.LR

print('SOURCE DISTRIBUTION: ', CONFIG.SOURCE)
print('TARGET DISTRIBUTION: ', CONFIG.TARGET)

# set up models dir
if CONFIG.MODELS_DIR is None:
    CONFIG.MODELS_DIR = ''.join(random.choices(string.ascii_lowercase, k=10))

#######################
### Datasets

target_sampler = StyleGANV2dataGenerator(
    CONFIG.TARGET, 
    'G_ema', 
    device=DEVICE, 
    data_parallel_ids=GPU_IDS if MULTI_GPU_FLAG else None)
CONFIG.RESOLUTION = target_sampler.z_dim

source_dataset_train = get_afhq_dataset(CONFIG.SOURCE, pix_range=(0., 1.), train=True)
source_dataset_test = get_afhq_dataset(CONFIG.SOURCE, pix_range=(0., 1.), train=False)
source_sampler_train = Dataset2Sampler(source_dataset_train, batch_size=10)
source_sampler_test = Dataset2Sampler(source_dataset_test, batch_size=10)

# replay buffer 
assert CONFIG.SAMPLE_BUFFER_TYPE == 'static'
X_train = torch.stack([source_dataset_train[i] for i in range(len(source_dataset_train))])

basic_noise_gen = target_sampler.latent_sampler

sample_buffer_instance = SampleBufferStatic(
    basic_noise_gen, X_train, device = DEVICE)

train_loader = DataLoaderWrapper(
    JointSampler(source_sampler_train, target_sampler.latent_sampler),
    CONFIG.BATCH_SIZE, n_batches=100)

test_loader = DataLoaderWrapper(
    JointSampler(source_sampler_test, target_sampler.latent_sampler),
    CONFIG.BATCH_SIZE, n_batches=20, store_dataset=True)

###############
### Model

if CONFIG.AMBIENT:
    class EgEOT_l2sq(EgEOT_l2sq_ambient_Mixin, ResNet_Dambient_withConfig):

        def __init__(self, custom_decoder, sample_buffer, config):
            super().__init__(custom_decoder, sample_buffer, config, custom_decoder, size=512)
else:
    class EgEOT_l2sq(EgEOT_l2sq_ambient_Mixin, FullyConnectedMLPwithConfig):

        def __init__(self, custom_decoder, sample_buffer, config, 
            hiddens=[1024, 512, 256],
            activation = lambda : nn.ReLU()):
            super().__init__(custom_decoder, sample_buffer, config, CONFIG.RESOLUTION, hiddens, 1, activation_gen=activation)

model = EgEOT_l2sq(
    target_sampler.latent2data,
    sample_buffer_instance,
    CONFIG)
if MULTI_GPU_FLAG:
    model.potential = DataParallelAttrAccess(
        model.potential, device_ids=GPU_IDS).to(DEVICE)

else:
    model.potential = model.potential.to(DEVICE)


optimizer = torch.optim.Adam(model.potential.parameters(), lr = CONFIG.LR, betas = CONFIG.ADAM_BETAS)

####################
### SCHEDULERS

all_models_path = os.path.join(THIS_SCRIPT_DIR, 'models') 
if not os.path.exists(all_models_path):
    os.makedirs(all_models_path)
save_models_path = os.path.join(all_models_path, CONFIG.MODELS_DIR)
os.makedirs(save_models_path)
CONFIG.store(os.path.join(save_models_path, 'config.pt'))
init_X_fixed_samples = torch.stack([source_dataset_test[i * 10] for i in range(10)]) #TODO!!!
if not MULTI_GPU_FLAG:
    init_X_fixed_samples = init_X_fixed_samples.to(DEVICE)

# lr scheduler
lr_mul_factors = get_changing_values_range(
    size = CONFIG.EPOCHS,
    val_init = 1.,
    val_fin = CONFIG.LR_FIN / CONFIG.LR,
    coldstart = CONFIG.LR_COLDSTART,
    coldfin = CONFIG.EPOCHS - CONFIG.LR_FREEZE_FINAL,
    progression='geom'
)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, 
    lambda epoch: lr_mul_factors[epoch]
)

schedulers = [
    TrainingSchedulerWandB_Mixin(use_wandb=CONFIG.USE_WANDB),
    TrainingSchedulerModelsSaver_Mixin(
        model=model, 
        save_models_path=save_models_path, 
        save_models_interval=100, 
        rewrite_saved_models=False),
    TrainingSchedulerWandB_EgEOT_Mixin(
        model=model, 
        config=CONFIG, 
        use_wandb=CONFIG.USE_WANDB,
        plot_images_interval=100,
        draw_replay_buffer_samples=True,
        is_image_space=False,
        init_X_fixed_samples=init_X_fixed_samples,
        target_data_transform=lambda x: target_sampler.latent2data(x).clip(0., 1.)), # replay buffer + train mode sampling
    TrainingSchedulerWandB_EgEOT_Mixin(
        model=model, 
        config=CONFIG, 
        use_wandb=CONFIG.USE_WANDB,
        plot_images_interval=1000,
        draw_replay_buffer_samples=False,
        is_image_space=False,
        train_mode_sampling = False,
        test_mode_sampling = True,
        init_X_fixed_samples=init_X_fixed_samples,
        target_data_transform=lambda x: target_sampler.latent2data(x).clip(0., 1.)), # test mode sampling
    TrainingSchedulerLR_Mixin(
        lr_scheduler=lr_scheduler)
]
clean_resources()

if CONFIG.USE_WANDB:
    wandb.init(name=args_dict['experiment'], project=WANDB_PROJECT_NAME, reinit=True, config=CONFIG)

# launch the training
train_model(
    model,
    train_loader, 
    test_loader, 
    epochs=CONFIG.EPOCHS,
    optimizer=optimizer,
    use_tqdm=True,
    loss_key='loss',
    conditional=True,
    scheduler=schedulers
)

if CONFIG.USE_WANDB:
    wandbdir = wandb.run.dir
    wandbid = wandb.run.id

    wandb.finish()

