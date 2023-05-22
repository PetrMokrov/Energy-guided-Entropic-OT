################################
# ## EVALUATE TRAINED EgEOT ## #
################################

import argparse
import torch
import torchvision.transforms as tr
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import json
import os
import numpy as np

from nets import VanillaNet, NonlocalNet
from utils import plot_ims
from utils import download_colored_mnist_data
from utils import plot_im_pairs
import wandb

WANDB_PROJECT_NAME = 'eot'

import os
import sys
sys.path.append('..')

DISCRETE_OT_DIR = '../src/discreteot'
sys.path.append(DISCRETE_OT_DIR)
from src.discreteot import DiscreteEOT_l2sq

parser = argparse.ArgumentParser(
    description='Evaluating longrun EgEOTs for CMnist2to3',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('experiment', help='experiment name')
parser.add_argument('--use_wandb', action='store_const', const=True, default=False)
parser.add_argument('--device', action='store', help='device (for NN training)', type=str, default='cuda:0')

args = parser.parse_args()

EXP_NAME = args.experiment
EXP_DIR = './out_data/{}/'.format(EXP_NAME)

BASIC_MODEL_PATH = './out_data/{}/checkpoints/'.format(EXP_NAME)
ch_files = [f for f in os.listdir(BASIC_MODEL_PATH) if os.path.isfile(os.path.join(BASIC_MODEL_PATH, f))]
for f in ch_files:
    assert f.endswith('.pth')

# json file with experiment config
CONFIG_FILE = './config_locker/{}.json'.format(EXP_NAME)
MODEL_NAME = max(ch_files)
print('run script for MODEL_NAME={}'.format(MODEL_NAME))
MODEL_PATH = './out_data/{}/checkpoints/{}'.format(EXP_NAME, MODEL_NAME)
FULL_DEVICE = args.device
USE_WANDB = args.use_wandb

BASIC_RESULT_SAVE_DIR = './out_eval'
RESULT_SAVE_DIR = BASIC_RESULT_SAVE_DIR + '/{}/'.format(EXP_NAME)
# make directory for saving results
if os.path.exists(BASIC_RESULT_SAVE_DIR):
    pass
else:
    os.makedirs(BASIC_RESULT_SAVE_DIR)

if os.path.exists(RESULT_SAVE_DIR):
    # prevents overwriting old experiment folders by accident
    raise RuntimeError('Folder "{}" already exists. Please use a different "RESULT_SAVE_DIR".'.format(RESULT_SAVE_DIR))
else:
    os.makedirs(RESULT_SAVE_DIR)


cost_grad_dict = {
    "l2sq": lambda x, y: y - x 
}

def l2sq_cost(x, y):
    assert x.shape == y.shape
    return 0.5 * torch.sum((x.view(x.size(0), -1) - y.view(y.size(0), -1)).pow(2), dim=-1)

cost_dict = {
    "l2sq": l2sq_cost
}


#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)


# set seed for cpu and CUDA, get device
# DEVICE SETTING
if FULL_DEVICE.startswith('cuda'):
    device = 'cuda'
    GPU_DEVICE = int(FULL_DEVICE.split(':')[1])
    torch.cuda.set_device(GPU_DEVICE)
else:
    device = 'cpu'

torch.manual_seed(config['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['seed'])


####################
# ## EVAL SETUP # ##
####################

print('Setting up EOT parameters...')
COST_GRAD = cost_grad_dict[config['cost']]
COST = cost_dict[config['cost']]
HREG = config['hreg']

print('Setting up network...')
# set up network
net_bank = {'vanilla': VanillaNet, 'nonlocal': NonlocalNet}
f = net_bank[config['net_type']](n_c=config['im_ch'])
# load saved weights
f.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage.cpu()))
# put net on device
f.to(device)

def compute_conditional_energy(f, x, y):
    '''
    f is the scaled lagrangian multiplier which was optimized following "config" parameters
    '''
    c = COST(x, y)
    f_y = f(y)
    assert c.shape == f_y.shape
    return (c / HREG + (2./ (config['epsilon'] ** 2)) * f_y)

print('Processing initial MCMC states...')
print('...downloading and preparing colored mnist data...')
src_data_name = 'MNISTcolored_2'
trg_data_name = 'MNISTcolored_3'

# download_colored_mnist_data(src_data_name)
# download_colored_mnist_data(trg_data_name)

data = {
    src_data_name: lambda path, func: datasets.ImageFolder(root=path, transform=func),
    trg_data_name: lambda path, func: datasets.ImageFolder(root=path, transform=func)
}
transform = tr.Compose([tr.Resize(config['im_sz']),
                        tr.CenterCrop(config['im_sz']),
                        tr.ToTensor(),
                        tr.Normalize(tuple(0.5*torch.ones(config['im_ch'])), tuple(0.5*torch.ones(config['im_ch'])))])
q_x = torch.stack([x[0] for x in data[src_data_name]('./data/' + src_data_name, transform)]).to(device)
q_y = torch.stack([x[0] for x in data[trg_data_name]('./data/' + trg_data_name, transform)]).to(device)

print('...set up initial MCMC states...')

MCMC_INIT = 'sourse_data' # we start the langevin dynamics from source data
BATCH_SIZE = 30 # number of initial colored images from X domain
N_PER = 6 # number of generated samples per each source sample

def sample_image_set(image_set):
    rand_inds = torch.randperm(image_set.shape[0])[-c]
    return image_set[rand_inds], rand_inds

# get a random sample of initial states from image bank
# X.repeat_interleave(n_rep, 0)
basic_inds = torch.arange(q_x.size(0) - BATCH_SIZE, q_x.size(0))
assert len(basic_inds) == BATCH_SIZE
__x_s_t_0 = q_x[basic_inds].repeat_interleave(N_PER, 0)
y_s_t_0, x_s_t_0 = __x_s_t_0.clone(), __x_s_t_0

if USE_WANDB:
    wandb.init(name=EXP_NAME, project=WANDB_PROJECT_NAME, reinit=True, config=config)
    print('WandB has initialized.')

plot_ims(RESULT_SAVE_DIR + 'initial_states.png', q_x[basic_inds], nrow=1)
if USE_WANDB:
    plot_ims(
        'dummy_name', q_x[basic_inds], 
        im_name='init Ys', n_step=0, use_wandb=USE_WANDB, nrow=1)
# plot_im_pairs(RESULT_SAVE_DIR + 'initial_states.png', q_x[basic_inds], y_s_t_0, invert=True, nrow=6)
config['batch_size'] = BATCH_SIZE * N_PER


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

# langevin equation without MH adjustment
def langevin_grad(epss, start_i=0):
    y_s_t = torch.autograd.Variable(y_s_t_0.clone(), requires_grad=True)

    # sampling records
    num_steps = len(epss)
    grads = torch.zeros(num_steps, config['batch_size'])
    ens = torch.zeros(num_steps, config['batch_size'])

    # iterative langevin updates of MCMC samples
    for ell, eps in enumerate(epss):
        en = compute_conditional_energy(f, x_s_t_0, y_s_t)
        ens[ell] = en.detach().cpu()
        grad = torch.autograd.grad(en.sum(), [y_s_t])[0]

        # y_s_t.data += grad + config['epsilon'] * torch.randn_like(y_s_t)
        y_s_t.data += - ((eps**2)/2) * grad + eps * torch.randn_like(y_s_t)
        # grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()
        grads[ell] = grad.view(grad.shape[0], -1).norm(dim=1).cpu()

        if ell == 0 or (ell + 1) % config['log_freq'] == 0 or (ell + 1) == num_steps:
            print('Step {} of {}.   Ave. En={:>14.9f}   Ave. Grad={:>14.9f}'.
                  format(ell+1, num_steps, ens[ell].mean(), grads[ell].mean()))
        if USE_WANDB:
            res_dict = {
                'ens': ens[ell].mean().item(),
                'grads': grads[ell].mean().item(),
                'eps': eps
            }
            wandb.log({'train': res_dict}, step=ell + start_i)
    return y_s_t.detach(), ens, grads


###################################
# ## SAMPLE FROM LEARNED MODEL ## #
###################################

# config['num_longrun_steps'] = 1000 #TODO

print('Sampling for {} Langevin steps.'.format(config['num_longrun_steps']))
epss = [config['epsilon'],]*config['num_longrun_steps']
y_s_t, en_record, grad_record = langevin_grad(epss)

plot_ims(RESULT_SAVE_DIR + 'interm_sample_states.png', y_s_t, nrow=N_PER)
if USE_WANDB:
    plot_ims(
        'dummy_name', y_s_t, 
        im_name='generated Ys', n_step=config['num_longrun_steps'], use_wandb=USE_WANDB, nrow=N_PER)

N_DECREASE_SAMPLES = 20000
# N_DECREASE_SAMPLES = 1000 #TODO
print('Sampling for {} Langevin steps.'.format(N_DECREASE_SAMPLES))
epss = np.geomspace(config['epsilon'], config['epsilon']/100., num=N_DECREASE_SAMPLES)
y_s_t_0 = y_s_t
y_s_t, en_record, grad_record = langevin_grad(epss, start_i=config['num_longrun_steps'])

# visualize initial and synthesized images
plot_ims(RESULT_SAVE_DIR + 'sample_states.png', y_s_t, nrow=N_PER)
if USE_WANDB:
    plot_ims(
        'dummy_name', y_s_t, 
        im_name='finally generated Ys', n_step=config['num_longrun_steps'] + N_DECREASE_SAMPLES, use_wandb=USE_WANDB, nrow=N_PER)

for i, ind in enumerate(basic_inds):
    _im = torch.cat((q_x[basic_inds][i].unsqueeze(0), y_s_t[i * N_PER:(i + 1) * N_PER]), dim=0)
    plot_ims(RESULT_SAVE_DIR + 'im{:02d}.png'.format(i + 1), _im, nrow=N_PER + 1)
# plot_im_pairs(RESULT_SAVE_DIR + 'sample_states.png', q_x[basic_inds], y_s_t, invert=True, nrow=6)
