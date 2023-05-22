import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

class StatsManager:

    @staticmethod
    def traverse(o, tree_types=(list, tuple, np.ndarray)):
        if isinstance(o, tree_types):
            for value in o:
                for subvalue in StatsManager.traverse(value, tree_types):
                    yield subvalue
        else:
            yield o
    
    def _request_presence(self, name):
        if not name in self.names:
            raise Exception("The name '{}' does not exist!".format(name))
    
    def _request_absence(self, name):
        if name in self.names:
            raise Exception("The name '{}' has already presented!".format(name))

    def __init__(self, *names):
        self.stats = {}
        self.names = list(names)
        for name in names:
            self.stats[name] = []
    
    def add_stat_names(self, *names):
        for name in names:
            self._request_absence(name)
            self.stats[name] = []
        self.names = self.names + list(names)

    def add_all(self, *vals):
        if len(vals) == 1:
            for name in self.names:
                if vals[0] is not None:
                    self.add(name, vals[0])
            return
        if len(vals) == len(self.names):
            for name, val in zip(self.names, vals):
                if val is not None:
                    self.add(name, val)
            return
        raise Exception('stats update is ambiguous')
    
    def mul(self, name, val):
        self._request_presence(name)
        self.stats[name][-1] *= val

    def add(self, name, val):
        self._request_presence(name)
        self.stats[name][-1] += val
    
    def upd_all(self, *vals):
        if len(vals) == 1:
            for name in self.names:
                if vals[0] is not None:
                    self.upd(name, vals[0])
            return
        if len(vals) == len(self.names):
            for name, val in zip(self.names, vals):
                if val is not None:
                    self.upd(name, val)
            return 
        raise Exception('stats update is ambiguous')
    
    def upd(self, name, val):
        if not name in self.names:
            self.add_stat_names(name)
        self.stats[name].append(val)
    
    def get(self, name):
        return self.stats[name]
    
    def draw(self, axs, names=None):
        axs_list = list(self.traverse(axs))
        if names is None:
            names = self.names
        for i, name in enumerate(names):
            axs_list[i].plot(self.get(name))
            axs_list[i].set_title(name)
    
    def reset(self):
        for name in self.stats.keys():
            self.stats[name] = []

class StatsManagerDrawScheduler:

    def __init__(self, SM, nrows, ncols, figsize, epoch_freq=1):
        self.SM = SM
        self.ncols = ncols
        self.nrows = nrows
        self.figsize = figsize
        self.epoch_freq = epoch_freq
        self.curr_epoch = 0

    def draw(self):
        clear_output(wait=True)
        fig, axes = plt.subplots(
            self.nrows, self.ncols, figsize=self.figsize)
        self.SM.draw(axes)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def epoch(self):
        self.curr_epoch += 1
        if self.curr_epoch == self.epoch_freq:
            self.draw()
            self.curr_epoch = 0
    