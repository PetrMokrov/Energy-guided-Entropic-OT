# Energy-guided-EntropicOT

This repository contains code to reproduce the experiments from our work [https://arxiv.org/abs/2304.06094](https://arxiv.org/abs/2304.06094)

## Citation

If you find this repository or the ideas presented in our paper useful, please consider citing our paper.

```
@misc{mokrov2023energyguided,
      title={Energy-guided Entropic Neural Optimal Transport}, 
      author={Petr Mokrov and Alexander Korotin and Evgeny Burnaev},
      year={2023},
      eprint={2304.06094},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Experiments 

Below, we give the instructions how to launch the experiments from our manuscript

### Toy 2D experiment (2D Gaussian to Swissroll)

The reproduction code could be found in

```
./notebooks/EgEOT_2D_Gauss2Swissroll_training.ipynb
./notebooks/EgEOT_2D_Gauss2Swissroll_image4paper.ipynb
```

The first notebook actually trains our `EgEOT` models for *Gaussian*$\rightarrow$*Swissroll* setup, while the second notebook reproduces the picture from our article based on pretrained models

### Gaussian to Gaussian

Our obtained $BW-UVP$ results could be achieved by launching the script `./scripts/egeot_gauss2gauss_train.py` with the corresponding parameters. In particular, the following command run the experiment for dimension $D=64$, entropic regularization coefficient $\varepsilon = 0.1$:
```bash
cd ./scripts
python3 egeot_gauss2gauss_train.py '<exp_name_you_want>' --eps 0.1 --dim 64 --use_wandb --device 'cuda:<number>'
```
Note that we use `wandb` ([link](https://wandb.ai/site)) dashboard system when launching our experiments. The practitioners are expected to use `wandb` too. 

### ColoredMnist 2 to 3

The code is located in `./mnist2to3` directory. The following two scripts are used for training and evaluating (generating images) correspondingly:
```
./mnist2to3/train_ot_data.py
./mnist2to3/eval_ot.py
```
The configs for the experiments are located in `./mnist2to3/config_locker` folder. For each entropic regularization coefficient $\varepsilon = h = 1, 0.1, 0.01$ we look over:

* number of Langevin steps when training $s = 500, 1000, 2000$

* different noise levels $n$ used to slightly noise the reference data for stability

#### Training

In order to run the experiment for, e.g., $h = 0.01, s = 1000, n = 0.1$, use the following commands:
```bash
cd ./mnist2to3
python3 train_ot_data.py 'mc2to3_h0.01_s1000_n0.1' --device 'cuda:<number>' --use_wandb
```

The images and checkpoints will be saved in the corresponding subdirectories of `./mnist2to3/out_data/mc2to3_h0.01_s1000_n0.1` directory.

#### Evaluation

Given pretrained models for a particular set of parameters $h, s, n$, e.g., $h = 0.01, s = 1000, n = 0.1$, one can evaluate these models by running:
```bash
cd ./mnist2to3
python3 eval_ot.py 'mc2to3_h0.01_s1000_n0.1' --device 'cuda:<number>' --use_wandb
```

The results could be found in `./mnist2to3/out_eval/mc2to3_h0.01_s1000_n0.1`. 
Note, that `eval_ot.py` evaluates the checkpoint with the largest number from `./mnist2to3/out_data/mc2to3_h0.01_s1000_n0.1`.





