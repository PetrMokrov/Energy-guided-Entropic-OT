##############################################
# ## TRAIN EgEOT USING Colored MNIST 2to3 ## #
##############################################

import argparse
import torch
import torchvision.transforms as tr
import torchvision.datasets as datasets
import json
import os
from nets import VanillaNet, NonlocalNet
from utils import plot_ims, plot_diagnostics
from utils import download_colored_mnist_data
from utils import plot_im_pairs
from utils import steps_counter
import wandb

WANDB_PROJECT_NAME = 'eot'

import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append('..')

DISCRETE_OT_DIR = '../src/discreteot'
sys.path.append(DISCRETE_OT_DIR)
from src.discreteot import DiscreteEOT_l2sq

# directory for experiment results
parser = argparse.ArgumentParser(
    description='Training longrun EgEOTs for CMnist2to3',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# genereal settings

parser.add_argument('experiment', help='experiment name')
parser.add_argument('--use_wandb', action='store_const', const=True, default=False)
parser.add_argument('--device', action='store', help='device (for NN training)', type=str, default='cuda:0')
args = parser.parse_args()

EXP_NAME = args.experiment
EXP_DIR = './out_data/{}/'.format(EXP_NAME)
# json file with experiment config
# CONFIG_FILE = './config_locker/flowers_convergent.json'
CONFIG_FILE = './config_locker/{}.json'.format(EXP_NAME)
FULL_DEVICE = args.device
USE_WANDB = args.use_wandb

cost_grad_dict = {
    "l2sq": lambda x, y: y - x 
}


#######################
# ## INITIAL SETUP ## #
#######################

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)

# make directory for saving results
if os.path.exists(EXP_DIR):
    # prevents overwriting old experiment folders by accident
    raise RuntimeError('Folder "{}" already exists. Please use a different "EXP_DIR".'.format(EXP_DIR))
else:
    os.makedirs(EXP_DIR)
    for folder in ['checkpoints', 'shortrun', 'longrun', 'plots', 'code']:
        os.mkdir(EXP_DIR + folder)

# save copy of code in the experiment folder
def save_code():
    def save_file(file_name):
        file_in = open('./' + file_name, 'r')
        file_out = open(EXP_DIR + 'code/' + os.path.basename(file_name), 'w')
        for line in file_in:
            file_out.write(line)
    for file in ['train_ot_data.py', 'nets.py', 'utils.py', CONFIG_FILE]:
        save_file(file)
save_code()

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


########################
# ## TRAINING SETUP # ##
########################
print('Setting up EgEOT parameters...')
COST_GRAD = cost_grad_dict[config['cost']]
HREG = config['hreg']

print('Setting up network and optimizer...')
# set up network
net_bank = {'vanilla': VanillaNet, 'nonlocal': NonlocalNet}
f = net_bank[config['net_type']](n_c=config['im_ch']).to(device)
# set up optimizer
optim_bank = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
if config['optimizer_type'] == 'sgd' and config['epsilon'] > 0:
    # scale learning rate according to langevin noise for invariant tuning
    config['lr_init'] *= (config['epsilon'] ** 2) / 2
    config['lr_min'] *= (config['epsilon'] ** 2) / 2
optim = optim_bank[config['optimizer_type']](f.parameters(), lr=config['lr_init'])

# print('Processing data...')
# # make tensor of training data
# download_colored_mnist_data('MNISTcolored_2')
# download_colored_mnist_data('MNISTcolored_3')

src_data_name = 'MNISTcolored_2'
trg_data_name = 'MNISTcolored_3'

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

# initialize persistent images from noise (one persistent image for each data image)
# s_t_0 is used when init_type == 'persistent' in sample_s_t()
s_t_0 = 2 * torch.rand_like(q_x) - 1

# sample batch from given array of images
def sample_image_set(image_set, size=config['batch_size']):
    rand_inds = torch.randperm(image_set.shape[0])[0:size]
    return image_set[rand_inds], rand_inds

################# DOT for init
def solve_dot(X, Y, numitermax=10000, verbose=False):
    DOT_DTYPE='torch64'
    DOT_NUMITERMAX = numitermax
    DOT_VERBOSE = verbose
    discr_eot = DiscreteEOT_l2sq(
        device=device, 
        verbose=DOT_VERBOSE, 
        numItermax=DOT_NUMITERMAX, 
        dtype=DOT_DTYPE
    ).solve(
        X.view(X.size(0), -1), 
        Y.view(Y.size(0), -1),
        config['hreg']
    )
    x_inds = torch.arange(X.size(0))
    y_inds = discr_eot.sample_by_indices(x_inds, return_indices=True)
    y_image_subset = Y[y_inds]
    return X, y_image_subset, (x_inds, y_inds)

if config['shortrun_init'] == 'persistentDOT':
    SC = steps_counter(s0=config['pDOT_update_step'], s1=1)


################################
# ## FUNCTIONS FOR SAMPLING ## #
################################

# sample positive images from dataset distribution q_y (add noise to ensure min sd is at least langevin noise sd)
def sample_q_y():
    x_q_y = sample_image_set(q_y)[0]
    return x_q_y + config['data_epsilon'] * torch.randn_like(x_q_y)

# initialize and update images with langevin dynamics to obtain samples from finite-step MCMC distribution s_t
def sample_s_t(L, init_type, update_s_t_0=True):
    # get initial mcmc states for langevin updates ("persistent", "data", "uniform", or "gaussian")
    def sample_s_t_0():
        '''
        returns (y_samples, x_samples, indices)
        '''
        if init_type == 'persistent':
            y_image_subset, rand_inds = sample_image_set(s_t_0)
            return y_image_subset, q_x[rand_inds], rand_inds
        elif init_type == 'DOT':
            X, _ = sample_image_set(q_x)
            Y, _ = sample_image_set(q_y)
            x_image_subset, y_image_subset, _ = solve_dot(X, Y, numitermax=2000)
            return y_image_subset, x_image_subset, None
        elif init_type == 'source_data':
            x_image_subset, inds = sample_image_set(q_x)
            return x_image_subset.clone().detach(), x_image_subset, (inds,)
        elif init_type == 'target_data':
            x_image_subset, inds = sample_image_set(q_x)
            y_image_subset, y_inds = sample_image_set(q_y)
            return y_image_subset, x_image_subset, (inds, y_inds)
        elif init_type == 'persistentDOT':
            y_image_subset, rand_inds = sample_image_set(s_t_0)
            if next(SC):
                X, x_inds = sample_image_set(q_x, size=1000)
                Y, _ = sample_image_set(q_y, size=1000)
                X, Y_dot, _ = solve_dot(X, Y, numitermax=10000)
                s_t_0[x_inds] = Y_dot
            return y_image_subset, q_x[rand_inds], rand_inds
        elif init_type == 'uniform':
            x_image_subset, _ = sample_image_set(q_x)
            noise_image = 2 * torch.rand([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']]) - 1
            return noise_image.to(device), x_image_subset, None
        elif init_type == 'gaussian':
            x_image_subset, _ = sample_image_set(q_x)
            noise_image = torch.randn([config['batch_size'], config['im_ch'], config['im_sz'], config['im_sz']])
            return noise_image.to(device), x_image_subset, None
        else:
            raise RuntimeError('Invalid method for "init_type" (use "persistent", "data", "uniform", or "gaussian")')

    # initialize MCMC samples
    y_s_t_0, x_s_t_0, s_t_0_inds = sample_s_t_0()

    # iterative langevin updates of MCMC samples
    y_s_t = torch.autograd.Variable(y_s_t_0.clone(), requires_grad=True)
    r_s_t = torch.zeros(1).to(device)  # variable r_s_t (Section 3.2) to record average gradient magnitude
    for ell in range(L):
        f_prime = torch.autograd.grad(f(y_s_t).sum(), [y_s_t])[0]
        grad_cost_coeff = (config['epsilon'] ** 2)/(2. * HREG)
        y_s_t.data += - f_prime - grad_cost_coeff * COST_GRAD(x_s_t_0 , y_s_t.data) + config['epsilon'] * torch.randn_like(y_s_t)
        r_s_t += f_prime.view(f_prime.shape[0], -1).norm(dim=1).mean()

    if init_type == 'persistent' and update_s_t_0:
        # update persistent image bank
        s_t_0.data[s_t_0_inds] = y_s_t.detach().data.clone()

    return y_s_t.detach(), x_s_t_0, r_s_t.squeeze() / L


#######################
# ## TRAINING LOOP ## #
#######################

# containers for diagnostic records (see Section 3)
d_s_t_record = torch.zeros(config['num_train_iters']).to(device)  # energy difference between positive and negative samples
r_s_t_record = torch.zeros(config['num_train_iters']).to(device)  # average image gradient magnitude along Langevin path

if USE_WANDB:
    wandb.init(name=EXP_NAME, project=WANDB_PROJECT_NAME, reinit=True, config=config)
    print('WandB has initialized.')

print('Training has started.')
for i in range(config['num_train_iters']):
    # obtain positive and negative samples
    samp_q_y = sample_q_y()
    y_s_t, x_s_t, r_s_t = sample_s_t(L=config['num_shortrun_steps'], init_type=config['shortrun_init'])

    # calculate ML computational loss d_s_t (Section 3) for data and shortrun samples
    d_s_t = f(samp_q_y).mean() - f(y_s_t).mean()
    if config['epsilon'] > 0:
        # scale loss with the langevin implementation
        d_s_t *= 2 / (config['epsilon'] ** 2)
    # stochastic gradient ML update for model weights
    optim.zero_grad()
    d_s_t.backward()
    optim.step()

    # record diagnostics
    d_s_t_record[i] = d_s_t.detach().data
    r_s_t_record[i] = r_s_t

    # anneal learning rate
    for lr_gp in optim.param_groups:
        lr_gp['lr'] = max(config['lr_min'], lr_gp['lr'] * config['lr_decay'])

    # update wandb data
    if USE_WANDB:
        res_dict = {
            'd_s_t': d_s_t.detach().data,
            'r_s_t': r_s_t
        }
        wandb.log({'train': res_dict}, step=i)


    # print and save learning info
    if (i + 1) == 1 or (i + 1) % config['log_freq'] == 0:
        print('{:>6d}   d_s_t={:>14.9f}   r_s_t={:>14.9f}'.format(i+1, d_s_t.detach().data, r_s_t))
        # visualize synthesized images
        plot_im_pairs(
            EXP_DIR + 'shortrun/' + 'pairs_x->y_s_t_{:>06d}.png'.format(i+1), x_s_t, y_s_t, 
            im_name='pairs x->y, shortrun, pbuff init', n_step=i, use_wandb=USE_WANDB)

        if config['shortrun_init'] == 'persistent':
            plot_ims(
                EXP_DIR + 'shortrun/' + 'y_s_t_0_{:>06d}.png'.format(i+1), s_t_0[0:config['batch_size']], 
                im_name='Ys from pbuff', n_step=i, use_wandb=USE_WANDB)
        # save network weights
        torch.save(f.state_dict(), EXP_DIR + 'checkpoints/' + 'net_{:>06d}.pth'.format(i+1))
        # plot diagnostics for energy difference d_s_t and gradient magnitude r_t
        if (i + 1) > 1:
            plot_diagnostics(i, d_s_t_record, r_s_t_record, EXP_DIR + 'plots/')

    # sample longrun chains to diagnose model steady-state
    if config['log_longrun'] and (i+1) % config['log_longrun_freq'] == 0:
        print('{:>6d}   Generating long-run samples. (L={:>6d} MCMC steps)'.format(i+1, config['num_longrun_steps']))
        for init_type in ['DOT', 'persistent', 'uniform', 'source_data', 'target_data']:
            y_p_theta, x_p_theta, _ = sample_s_t(L=config['num_longrun_steps'], init_type=init_type, update_s_t_0=False)
            plot_im_pairs(
                EXP_DIR + 'longrun/' + 'longrun_{}_{:>06d}.png'.format(init_type, i+1), x_p_theta, y_p_theta, 
                im_name='pairs x->y, longrun, {} init'.format(init_type), n_step=i, use_wandb=USE_WANDB)
            print('{:>6d}   Long-run samples for init {} saved.'.format(i+1, init_type))
