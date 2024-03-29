# Energy-guided-EntropicOT

This repository contains **code** for the experiments of ICLR'2024 paper [*Energy-guided Entropic Neural Optimal Transport*](https://openreview.net/forum?id=d6tUsZeVs7) by [Petr Mokrov](https://scholar.google.com/citations?user=CRsi4IkAAAAJ&hl=en), [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Alexander Kolesov](https://scholar.google.com/citations?user=WyAI_wUAAAAJ&hl=en), [Nikita Gushchin](https://scholar.google.com/citations?user=UaRTbNoAAAAJ&hl=en), [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=en). We propose to solve Entropic Optimal Transport problem in continuous space with help of Energy-Based Models. Our experiments showcase the performance of our method in different illustrative and image-domain scenarious.

<p align="center"><img src="teaser/afhq_egeot.png" width="800" /></p>

## Citation

If you find this repository or the ideas presented in our paper useful, please consider citing our paper.

```
@inproceedings{
      mokrov2024energyguided,
      title={Energy-guided Entropic Neural Optimal Transport},
      author={Petr Mokrov and Alexander Korotin and Alexander Kolesov and Nikita Gushchin and Evgeny Burnaev},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://openreview.net/forum?id=d6tUsZeVs7}
}
```

## Experiments 

Below, we give the instructions how to launch the experiments from our manuscript.

We assume, that the path to this project directory is `<PROJDIR>`, e.g., this readme file has path `<PROJDIR>/README.md`

### Toy 2D experiment (2D Gaussian to Swissroll)

The reproduction code could be found in

```
<PROJDIR>/notebooks/EgEOT_2D_Gauss2Swissroll_training.ipynb
<PROJDIR>/notebooks/EgEOT_2D_Gauss2Swissroll_image4paper.ipynb
```

The first notebook actually trains our `EgEOT` models for *Gaussian*$\rightarrow$*Swissroll* setup, while the second notebook reproduces the picture from our article based on pretrained models

### Gaussian to Gaussian

Our obtained $BW-UVP$ results could be achieved by launching the script `./scripts/egeot_gauss2gauss_train.py` with the corresponding parameters. In particular, the following command run the experiment for dimension $D=64$, entropic regularization coefficient $\varepsilon = 0.1$:
```bash
cd <PROJDIR>/scripts
python3 egeot_gauss2gauss_train.py '<exp_name_you_want>' --eps 0.1 --dim 64 --use_wandb --device 'cuda:<number>'
```
Note that we use `wandb` ([link](https://wandb.ai/site)) dashboard system when launching our experiments. The practitioners are expected to use `wandb` too. 

### ColoredMnist 2 to 3

The code is located in `<PROJDIR>/mnist2to3` directory. The following two scripts are used for training and evaluating (generating images) correspondingly:
```
<PROJDIR>/mnist2to3/train_ot_data.py
<PROJDIR>/mnist2to3/eval_ot.py
```
The configs for the experiments are located in `<PROJDIR>/mnist2to3/config_locker` folder. For each entropic regularization coefficient $\varepsilon = h = 1, 0.1, 0.01$ we look over:

* number of Langevin steps when training $s = 500, 1000, 2000$

* different noise levels $n$ used to slightly noise the reference data for stability

#### Downloadin ColoredMNIST dataset

```bash
cd <PROJDIR>/mnist2to3
python3 download_data.py
```

#### Training

In order to run the experiment for, e.g., $h = 0.01, s = 1000, n = 0.1$, use the following commands:
```bash
cd <PROJDIR>/mnist2to3
python3 train_ot_data.py 'mc2to3_h0.01_s1000_n0.1' --device 'cuda:<number>' --use_wandb
```

The images and checkpoints will be saved in the corresponding subdirectories of `<PROJDIR>/mnist2to3/out_data/mc2to3_h0.01_s1000_n0.1` directory.

#### Evaluation

Given pretrained models for a particular set of parameters $h, s, n$, e.g., $h = 0.01, s = 1000, n = 0.1$, one can evaluate these models by running:
```bash
cd <PROJDIR>/mnist2to3
python3 eval_ot.py 'mc2to3_h0.01_s1000_n0.1' --device 'cuda:<number>' --use_wandb
```

The results could be found in `<PROJDIR>/mnist2to3/out_eval/mc2to3_h0.01_s1000_n0.1`. 
Note, that `eval_ot.py` evaluates the checkpoint with the largest number from `<PROJDIR>/mnist2to3/out_data/mc2to3_h0.01_s1000_n0.1`.

### AFHQ high-dimensional experiments

Before launching the experiment one need to prepare the dataset and pretrained StyleGAN2-ADA. 

#### Downloading AFHQ dataset

```bash
cd <PROJDIR>/src/datasets
bash thirdparty/download.sh
```

#### Setting up StyleGAN2-ADA model

Clone the official StyleGAN2-ADA [github](https://github.com/NVlabs/stylegan2-ada-pytorch):

```bash
cd <PROJDIR>/latentspace/thirdparty/
git clone https://github.com/NVlabs/stylegan2-ada-pytorch
```

Rename the downloaded repository:

```bash
mv stylegan2-ada-pytorch stylegan2_ada_pytorch
```

Download the pretrained StyleGAN2-ADA on AFHQ Dogs:

```bash
cd stylegan2_ada_pytorch
mkdir data && cd data
mkdir pretrained && cd pretrained
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl
```

##### Possible problems when launching pretrained StyleGAN2-ADA model

For the optimization purposes, the code from stylegan2 repository compiles some operations from scratch. For some hardware architectrues (in particular, for A100 servers with `sm_80` cuda capability) this compilation from scratch fails. One can avoid this problem by falling back to slow reference implementation. This can be done by slightly modifying the code for the following files: 
* `<PROJDIR>/latentspace/thirdparty/stylegan2_ada_pytorch/torch_utils/ops/bias_act.py`
* `<PROJDIR>/latentspace/thirdparty/stylegan2_ada_pytorch/torch_utils/ops/upfirdn2d.py`

In particular, one can substitute the implementation of `_init()` function from these files with:
```python
def _init():
      return False
```

#### Training

For launching the experiments use the script `<PROJDIR>/latentspace/scripts/egeot_afhq_stylegan_latent.py`

For running `Cat->Dog` experiment:

```bash
cd <PROJDIR>/latentspace/scripts/
python3 egeot_afhq_stylegan_latent.py 'your_experiment_name' --source 'cat' --target 'dog' --use_wandb --gpu_ids 0 1 2 3 # utilizes four gpus
```

Note, that the learning parameters assumes A100/V100 gpus. The code is tested on 4xA100.

