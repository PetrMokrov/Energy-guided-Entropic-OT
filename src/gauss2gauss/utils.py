import wandb
import gc
import torch

import sys
sys.path.append('../..')
from dgm_utils.scheduler import TrainingSchedulerGeneric

# eot benchmark for rotated gaussians
from src.gauss2gauss.dataset import (
    get_rotated_gaussian_dataset,
    get_rotated_gaussian_sampler,
    get_rotated_gaussian_benchmark_stats)
from src.gauss2gauss.metrics import compute_BW_UVP_with_gt_stats

from src.eot_utils import conditional_sample_from_EgEOT

VERY_BIG_NUMBER = 1000000.0

class _min_supporter:
    
    def __init__(self):
        self.value = VERY_BIG_NUMBER
    
    def upd(self, value):
        self.value = min(self.value, value)
    
    def get(self):
        return self.value


class TrainingScheduler_BW_UVP_Mixin(TrainingSchedulerGeneric):

    @property
    def min_marg_bw_uvp(self):
        if not hasattr(self, '_min_marg_bw_uvp'):
            return VERY_BIG_NUMBER
        return self._min_marg_bw_uvp

    @min_marg_bw_uvp.setter
    def min_marg_bw_uvp(self, value):
        self._min_marg_bw_uvp = min(value, self.min_marg_bw_uvp)

    @property
    def min_plan_bw_uvp(self):
        if not hasattr(self, '_min_plan_bw_uvp'):
            return VERY_BIG_NUMBER
        return self._min_plan_bw_uvp
    
    @min_plan_bw_uvp.setter
    def min_plan_bw_uvp(self, value):
        self._min_plan_bw_uvp = min(value, self.min_plan_bw_uvp)
    
    def sample_from_model_bw_uvp(self):
        return conditional_sample_from_EgEOT(
            self.model,
            self.config,
            self.estimate_bw_uvp_X_samples,
            1, # y_per_x
            train_mode=False,
            batch_size=self.config.BATCH_SIZE,
            to_cpu=True,
            images_flag=False
        )

    def __init__(
        self, 
        *args, 
        save_bw_uvp_interval=1,
        bw_uvp_apply_mode='test',
        bw_uvp_stats=None,
        **kwargs
    ):
        self.save_bw_uvp_interval = save_bw_uvp_interval
        self.estimate_bw_uvp_X_samples = self.extract_kwargs(
            kwargs, 'estimate_bw_uvp_X_samples').cpu()
        self.bw_uvp_apply_mode = bw_uvp_apply_mode
        assert self.bw_uvp_apply_mode in ['train', 'test']
        assert len(bw_uvp_stats) == 6
        self.bw_uvp_stats = {
            'mu_X': bw_uvp_stats[0],
            'mu_Y': bw_uvp_stats[1],
            'cov_X': bw_uvp_stats[2],
            'cov_Y': bw_uvp_stats[3],
            'mu_plan': bw_uvp_stats[4],
            'cov_plan': bw_uvp_stats[5]
        }
        super().__init__(*args, **kwargs)

    def compute_bw_uvp(self):
        eot_sample = self.sample_from_model_bw_uvp()
        marg_bw_uvp = compute_BW_UVP_with_gt_stats(
            eot_sample.numpy(),
            true_samples_mu=self.bw_uvp_stats['mu_Y'],
            true_samples_covariance=self.bw_uvp_stats['cov_Y']
        )
        assert self.estimate_bw_uvp_X_samples.shape == eot_sample.shape
        assert len(eot_sample.shape) == 2
        plan_sample = torch.cat([
            self.estimate_bw_uvp_X_samples, 
            eot_sample], dim=1)
        plan_bw_uvp = compute_BW_UVP_with_gt_stats(
            plan_sample.numpy(),
            true_samples_mu=self.bw_uvp_stats['mu_plan'],
            true_samples_covariance=self.bw_uvp_stats['cov_plan']
        )
        return marg_bw_uvp, plan_bw_uvp
    
    def compute_bw_uvp_and_update_losses(self, losses):
        marg_bw_uvp, plan_bw_uvp = self.compute_bw_uvp()
        losses['marg_bw_uvp'] = marg_bw_uvp
        losses['plan_bw_uvp'] = plan_bw_uvp
        self.min_marg_bw_uvp = marg_bw_uvp
        self.min_plan_bw_uvp = plan_bw_uvp
        losses['min_marg_bw_uvp'] = self.min_marg_bw_uvp
        losses['min_plan_bw_uvp'] = self.min_plan_bw_uvp

    def on_batch_train_end(self, epoch=None, batch=None, losses=None, data=None):
        if self.bw_uvp_apply_mode == 'test':
            pass
        else:
            self.compute_bw_uvp_and_update_losses(losses)

        super().on_batch_train_end(epoch, batch, losses, data)

    def on_epoch_eval_end(self, epoch=None, losses=None):
        if self.bw_uvp_apply_mode == 'train':
            pass
        else:
            self.compute_bw_uvp_and_update_losses(losses)

        super().on_epoch_eval_end(epoch, losses)
