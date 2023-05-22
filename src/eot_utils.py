import torch
import torch.nn as nn
import torch.autograd as autograd
from contextlib import contextmanager
from torchvision import utils as TVutils
import wandb
import gc
import torch

from src.utils import computePotGrad
from dgm_utils.scheduler import TrainingSchedulerWandB_Mixin
from dgm_utils.scheduler import TrainingSchedulerFID_IS_Mixin

def conditional_sample_from_EgEOT(
    model, 
    config, 
    X, 
    y_per_x, 
    train_mode=False, 
    batch_size=None, 
    to_cpu=True, 
    images_flag=True
):
    if images_flag:
        back_pix_val_transform = lambda x : (x - config.PIX_VAL_MIN) / (config.PIX_VAL_MAX - config.PIX_VAL_MIN)
    else:
        back_pix_val_transform = lambda x: x

    X_rep = X.repeat_interleave(y_per_x, 0)
    n_iterations = config.ENERGY_SAMPLING_ITERATIONS
    step_size = config.ENERGY_SAMPLING_STEP
    decay = config.LANGEVIN_DECAY
    if not train_mode:
        n_iterations = config.TEST_ENERGY_SAMPLING_ITERATIONS if \
            hasattr(config, 'TEST_ENERGY_SAMPLING_ITERATIONS') else config.ENERGY_SAMPLING_ITERATIONS
        step_size = config.TEST_ENERGY_SAMPLING_STEP if \
            hasattr(config, 'TEST_ENERGY_SAMPLING_STEP') else config.ENERGY_SAMPLING_STEP
        decay = config.TEST_LANGEVIN_DECAY if \
            hasattr(config, 'TEST_LANGEVIN_DECAY') else config.LANGEVIN_DECAY
    Y_init = model.sample_buffer.noise_gen.sample((X_rep.size(0),))

    if batch_size is None:
        batch_size = X_rep.size(0)

    Y_arr = []
    for i in range(0, len(X_rep), batch_size):
        start, end = i, min(i + batch_size, len(X_rep))
        _spl = model.sample(
            X_rep[start:end].to(Y_init.device),
            n_iterations=n_iterations,
            step_size=step_size,
            decay=decay,
            y_init=Y_init[start:end]).detach()
        if to_cpu:
            _spl = _spl.cpu()
        Y_arr.append(_spl)
    Y = torch.cat(Y_arr)
    if images_flag:
        Y.clamp_(config.PIX_VAL_MIN, config.PIX_VAL_MAX)
    im_shape = tuple(Y.shape[1:])
    Y = Y.unsqueeze(0).view(X.size(0), y_per_x, *im_shape).transpose(0, 1).reshape(-1, *im_shape)
    # gc.collect(); torch.cuda.empty_cache()

    return back_pix_val_transform(Y)

class TrainingSchedulerWandB_EgEOT_Mixin(TrainingSchedulerWandB_Mixin):

    def __init__(
        self, *args, 
        plot_images_interval=100,
        draw_replay_buffer_samples=True,
        draw_x_samples=10, 
        draw_y_samples_per_x=4,
        init_X_fixed_samples = None,
        init_X_sampler = None,
        train_mode_sampling = True,
        test_mode_sampling = False,
        **kwargs
    ):
        self.model, self.config, self.use_wandb = self.extract_kwargs(
            kwargs, 'model', 'config', 'use_wandb', del_names=False)
        self.init_X_fixed_samples = init_X_fixed_samples
        self.init_X_sampler = init_X_sampler
        self.draw_x_samples = draw_x_samples
        self.draw_y_samples_per_x = draw_y_samples_per_x
        self.plot_images_interval = plot_images_interval
        self.draw_replay_buffer_samples = draw_replay_buffer_samples
        self.sampling_modes = {
            'train': train_mode_sampling,
            'test': test_mode_sampling
        }
        super().__init__(*args, **kwargs)

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        back_pix_val_transform = lambda x : (x - self.config.PIX_VAL_MIN) / (self.config.PIX_VAL_MAX - self.config.PIX_VAL_MIN)
        
        if self.use_wandb:
            if self._steps_counter % self.plot_images_interval == 0:
                # samples from the replay buffer:
                if self.draw_replay_buffer_samples:
                    X, Y = self.model.sample_buffer.get(self.draw_x_samples)
                    SB_torch_grid = TVutils.make_grid(torch.cat([X, Y]), nrow=X.size(0), pad_value=1.)
                    SB_images = wandb.Image(SB_torch_grid, caption='top: X, bottom: Y')
                    wandb.log({"Replay Buffer samples": [SB_images,]}, step=self._steps_counter)

                # samples from the model given init_X_fixed_samples
                if self.init_X_fixed_samples is not None:
                    for mode, do_sampling in self.sampling_modes.items():
                        if not do_sampling:
                            continue

                        X = self.init_X_fixed_samples
                        Y = conditional_sample_from_EgEOT(
                            self.model, 
                            self.config,
                            X, self.draw_y_samples_per_x, 
                            train_mode=True if mode == 'train' else False, 
                            to_cpu=False,
                            batch_size=self.config.BATCH_SIZE
                        )
                        SB_torch_grid = TVutils.make_grid(
                            torch.cat([
                                back_pix_val_transform(X), 
                                Y]), nrow=X.size(0), pad_value=1.)
                        SB_images = wandb.Image(SB_torch_grid, caption='top row: X, bottom rows: Ys|X')
                        wandb.log({
                            "Random init Ys samples, fixed Xs, {} mode".format(mode): [SB_images,]
                        }, step=self._steps_counter)

                # samples from the model given random Xs
                if self.init_X_sampler is not None:
                    X = self.init_X_sampler.sample(self.draw_x_samples)
                    for mode, do_sampling in self.sampling_modes.items():
                        if not do_sampling:
                            continue

                        # X = self.init_X_sampler.sample(self.draw_x_samples)
                        Y = conditional_sample_from_EgEOT(
                            self.model, 
                            self.config,
                            X, self.draw_y_samples_per_x, 
                            train_mode=True if mode == 'train' else False,
                            to_cpu=False,
                            batch_size=self.config.BATCH_SIZE
                        )
                        SB_torch_grid = TVutils.make_grid(torch.cat(
                            [
                                back_pix_val_transform(X), 
                                Y
                            ]), nrow=X.size(0), pad_value=1.)
                        SB_images = wandb.Image(SB_torch_grid, caption='top row: X, bottom rows: Ys|X')
                        wandb.log({
                            "Random init Ys samples, random Xs, {} mode".format(mode): [SB_images,]
                        }, step=self._steps_counter)
                gc.collect(); torch.cuda.empty_cache()
        gc.collect(); torch.cuda.empty_cache()
        super().on_batch_train_end(epoch, batch, losses, data)

    def on_epoch_eval_end(self, epoch=None, losses=None):
        super().on_epoch_eval_end(epoch, losses)


class TrainingScheduler_EgEOT_FID_IS_Mixin(TrainingSchedulerFID_IS_Mixin):

    def sample_from_model(self, n_samples):
        return conditional_sample_from_EgEOT(
            self.model, 
            self.config,
            self.estimate_fid_is_X_samples,
            self.estimate_fid_is_y_samples_per_x,
            train_mode=False,
            batch_size=self.config.BATCH_SIZE,
            to_cpu=True
        )

    def __init__(
        self,
        *args, 
        estimate_fid_is_y_samples_per_x=1,
        **kwargs
    ):
        self.model, self.config = self.extract_kwargs(
            kwargs, 'model', 'config', del_names=False)
        self.estimate_fid_is_X_samples = self.extract_kwargs(
            kwargs, 'estimate_fid_is_X_samples').cpu()
        self.estimate_fid_is_y_samples_per_x = estimate_fid_is_y_samples_per_x
        # estimate_fid_is_n_samples
        if 'estimate_fid_is_n_samples' in kwargs.keys():
            raise Exception(
                "ambiguous values for 'estimate_fid_is_n_samples'!")
        kwargs['estimate_fid_is_n_samples'] = len(self.estimate_fid_is_X_samples) * self.estimate_fid_is_y_samples_per_x
        super().__init__(*args, **kwargs)


#####################
# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998

@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()