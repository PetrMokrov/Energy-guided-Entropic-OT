# download Oxford Flowers 102, plotting functions, and toy dataset

import torch as t
import torchvision.datasets as datasets
import torchvision as tv
from torch.utils.data import TensorDataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import shutil
sys.path.append('../..')
from torchvision.utils import save_image
import wandb


################################
# ## DOWNLOAD COLORED MNIST ## #
################################

def get_random_colored_images(images, pix_val_range=(-1., 1.), seed=0x000000):
    np.random.seed(seed)
    
    pix_val_transform = lambda x : (pix_val_range[1] - pix_val_range[0]) * x + pix_val_range[0]
    back_pix_val_transform = lambda x : (x - pix_val_range[0]) / (pix_val_range[1] - pix_val_range[0])

    images = back_pix_val_transform(images)
    size = images.shape[0]
    colored_images = []
    hues = 360*np.random.rand(size)
    
    for V, H in zip(images, hues):
        V_min = 0
        
        a = (V - V_min)*(H%60)/60
        V_inc = a
        V_dec = V - a
        
        colored_image = t.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H/60) % 6
        
        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec
        
        colored_images.append(colored_image)
        
    colored_images = t.stack(colored_images, dim = 0)
    colored_images = pix_val_transform(colored_images)
    
    return colored_images

def load_cmnist_dataset(name, path, batch_size=64, shuffle=True, device='cuda', pix_val_range=(-1., 1.), seed=0x000000):

    pix_val_transform = lambda x : (pix_val_range[1] - pix_val_range[0]) * x + pix_val_range[0]
    
    assert name.startswith("MNIST")
    
    # In case of using certain classe from the MNIST dataset you need to specify them by writing in the next format "MNIST_{digit}_{digit}_..._{digit}"
    transform = tv.transforms.Compose([
        tv.transforms.Resize((32, 32)),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(pix_val_transform)
    ])

    dataset_name = name.split("_")[0]
    is_colored = dataset_name[-7:] == "colored"

    classes = [int(number) for number in name.split("_")[1:]]
    if not classes:
        classes = [i for i in range(10)]

    train_set = datasets.MNIST(
        path, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(
        path, train=False, transform=transform, download=True)

    train_test = []

    for dataset in [train_set, test_set]:
        data = []
        labels = []
        for k in range(len(classes)):
            data.append(t.stack(
                [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == classes[k]],
                dim=0
            ))
            labels += [k]*data[-1].shape[0]
        data = t.cat(data, dim=0)
        data = data.reshape(-1, 1, 32, 32)
        labels = t.tensor(labels)

        if is_colored:
            data = get_random_colored_images(data, pix_val_range, seed=seed)

        train_test.append(TensorDataset(data, labels))

    train_set, test_set = train_test
    return train_set, test_set

def download_colored_mnist_data(name):

    first_feature = lambda x : x[0] if isinstance(x, (list,tuple)) else x

    dataset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/{}/'.format(name))
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    else:
        return
    raw_folder = os.path.join(dataset_folder, '__raw')
    ims_folder = os.path.join(dataset_folder, 'ims')
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)
    os.makedirs(ims_folder)
    train_dataset, test_dataset = load_cmnist_dataset(
        name, raw_folder, batch_size=64, shuffle=False, device='cpu', pix_val_range=(0., 1.))

    i = 0
    for dataset in [train_dataset, test_dataset]:
        for im in dataset:
            im_file_name = os.path.join(ims_folder, 'im_{:>06d}.png'.format(i))
            save_image(first_feature(im), im_file_name)
            i += 1

    shutil.rmtree(raw_folder)


##################
# ## PLOTTING ## #
##################

# visualize negative samples synthesized from energy
def plot_ims(p, x, n_step=None, im_name='dummy name', use_wandb=False, nrow=None, invert=False): 
    x = t.clamp(x, -1., 1.)
    if invert:
        x = 1. - x
    if nrow is None:
        nrow = int(x.shape[0] ** 0.5)
    pad_value = 1. if invert else 0.
    if not use_wandb:
        tv.utils.save_image(x, p, normalize=True, nrow=nrow, pad_value=pad_value)
    else:
        SB_torch_grid = tv.utils.make_grid(x, nrow=nrow, pad_value=pad_value, normalize=True)
        SB_images = wandb.Image(SB_torch_grid, caption='Xs')
        wandb.log({im_name: [SB_images,]}, step=n_step)

def plot_im_pairs(p, x, y, n_step=None, im_name='dummy name', use_wandb=False, nrow=None, invert=False):
    if nrow is None:
        nrow = int(x.shape[0] ** 0.5)
    assert x.shape == y.shape
    im_shape = tuple(x.shape[1:])
    to_draw = t.clamp(t.cat([x.unsqueeze(1), y.unsqueeze(1)], 1).view(-1, *im_shape), -1., 1.)
    if invert:
        to_draw = 1. - to_draw
    pad_value = 1. if invert else 0.
    if not use_wandb:
        tv.utils.save_image(to_draw, p, normalize=True, nrow=nrow, pad_value=pad_value)
    else:
        SB_torch_grid = tv.utils.make_grid(to_draw, nrow=nrow, pad_value=pad_value, normalize=True)
        SB_images = wandb.Image(SB_torch_grid, caption='first: X, second: Y')
        wandb.log({im_name: [SB_images,]}, step=n_step)
    

# plot diagnostics for learning
def plot_diagnostics(batch, en_diffs, grad_mags, exp_dir, fontsize=10):
    # axis tick size
    matplotlib.rc('xtick', labelsize=6)
    matplotlib.rc('ytick', labelsize=6)
    fig = plt.figure()

    def plot_en_diff_and_grad_mag():
        # energy difference
        ax = fig.add_subplot(221)
        ax.plot(en_diffs[0:(batch+1)].data.cpu().numpy())
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Energy Difference', fontsize=fontsize)
        ax.set_xlabel('batch', fontsize=fontsize)
        ax.set_ylabel('$d_{s_t}$', fontsize=fontsize)
        # mean langevin gradient
        ax = fig.add_subplot(222)
        ax.plot(grad_mags[0:(batch+1)].data.cpu().numpy())
        ax.set_title('Average Langevin Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('batch', fontsize=fontsize)
        ax.set_ylabel('$r_{s_t}$', fontsize=fontsize)

    def plot_crosscorr_and_autocorr(t_gap_max=2000, max_lag=15, b_w=0.35):
        t_init = max(0, batch + 1 - t_gap_max)
        t_end = batch + 1
        t_gap = t_end - t_init
        max_lag = min(max_lag, t_gap - 1)
        # rescale energy diffs to unit mean square but leave uncentered
        en_rescale = en_diffs[t_init:t_end] / t.sqrt(t.sum(en_diffs[t_init:t_end] * en_diffs[t_init:t_end])/(t_gap-1))
        # normalize gradient magnitudes
        grad_rescale = (grad_mags[t_init:t_end]-t.mean(grad_mags[t_init:t_end]))/t.std(grad_mags[t_init:t_end])
        # cross-correlation and auto-correlations
        cross_corr = np.correlate(en_rescale.cpu().numpy(), grad_rescale.cpu().numpy(), 'full') / (t_gap - 1)
        en_acorr = np.correlate(en_rescale.cpu().numpy(), en_rescale.cpu().numpy(), 'full') / (t_gap - 1)
        grad_acorr = np.correlate(grad_rescale.cpu().numpy(), grad_rescale.cpu().numpy(), 'full') / (t_gap - 1)
        # x values and indices for plotting
        x_corr = np.linspace(-max_lag, max_lag, 2 * max_lag + 1)
        x_acorr = np.linspace(0, max_lag, max_lag + 1)
        t_0_corr = int((len(cross_corr) - 1) / 2 - max_lag)
        t_0_acorr = int((len(cross_corr) - 1) / 2)

        # plot cross-correlation
        ax = fig.add_subplot(223)
        ax.bar(x_corr, cross_corr[t_0_corr:(t_0_corr + 2 * max_lag + 1)])
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Cross Correlation of Energy Difference\nand Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('lag', fontsize=fontsize)
        ax.set_ylabel('correlation', fontsize=fontsize)
        # plot auto-correlation
        ax = fig.add_subplot(224)
        ax.bar(x_acorr-b_w/2, en_acorr[t_0_acorr:(t_0_acorr + max_lag + 1)], b_w, label='en. diff. $d_{s_t}$')
        ax.bar(x_acorr+b_w/2, grad_acorr[t_0_acorr:(t_0_acorr + max_lag + 1)], b_w, label='grad. mag. $r_{s_t}}$')
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Auto-Correlation of Energy Difference\nand Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('lag', fontsize=fontsize)
        ax.set_ylabel('correlation', fontsize=fontsize)
        ax.legend(loc='upper right', fontsize=fontsize-4)

    # make diagnostic plots
    plot_en_diff_and_grad_mag()
    plot_crosscorr_and_autocorr()
    # save figure
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    plt.savefig(os.path.join(exp_dir, 'diagnosis_plot.pdf'), format='pdf')
    plt.close()


def steps_counter(s0, s1, res0=False, res1=True):
    assert res0 != res1
    curr_step = 0
    steps_passed = 0
    res_mapping = [res0, res1]
    while True:
        steps_passed += 1
        if curr_step == 0:
            if steps_passed > s0:
                curr_step = 1
                steps_passed = 1
        elif curr_step == 1:
            if steps_passed > s1:
                curr_step = 0
                steps_passed = 1
        yield res_mapping[curr_step]
