# Group-Equivariant GANs (ICLR 2021)
### [Paper](https://arxiv.org/abs/2005.01683) | [Video (coming soon)]() 
![Samples Collage](https://pbs.twimg.com/media/EXQ84JnXQAgBWFw?format=jpg&name=large)

Barebones (unofficial) Tensorflow 2 implementation of *Group Equivariant Generative Adversarial Networks* presented at ICLR 2021.

```
@inproceedings{
	dey2021group,
	title={Group Equivariant Generative Adversarial Networks},
	author={Neel Dey and Antong Chen and Soheil Ghafurian},
	booktitle={International Conference on Learning Representations},
	year={2021},
	url={https://openreview.net/forum?id=rgFNuJHHXv}
}
```

## Dependencies
Assuming that you use conda, you can set up and use an environment with all required dependencies using:
```bash
conda create --name equiv-gan
conda install tensorflow-gpu=2.2
pip install tensorflow-addons==0.11.2
pip install git+https://github.com/neel-dey/tf2-GrouPy#egg=GrouPy -e git+https://github.com/neel-dey/tf-keras-gcnn.git#egg=keras_gcnn

conda activate equiv-gan
```

## Usage
This repository is geared towards the main experiments in the paper and thus the architectures and data loading schemes are specific to those. Adding support for your own architectures and data loaders should be pretty straightforward.
 
A sample training call for CIFAR10 might be:
```python
python train_script.py \
--epochs 600 --batchsize 64 --name samplecifar --gp_wt 0.01 \
--lr_g 2e-4 --lr_d 2e-4 --g_arch p4_cifar10 --d_arch p4_cifar10 \
--d_updates 4 --dataset cifar10
```

where the full CLI corresponds to:

```bash
usage: train_script.py [-h] [--epochs EPOCHS] [--batchsize BATCHSIZE]
                       [--name NAME] [--latent_dim LATENT_DIM] [--lr_g LR_G]
                       [--beta1_g BETA1_G] [--beta2_g BETA2_G] [--lr_d LR_D]
                       [--beta1_d BETA1_D] [--beta2_d BETA2_D] [--gp_wt GP_WT]
                       [--ckpt_freq CKPT_FREQ] [--resume_ckpt RESUME_CKPT]
                       [--rng RNG] [--g_arch G_ARCH] [--d_arch D_ARCH]
                       [--loss LOSS] [--d_updates D_UPDATES]
                       [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of training epochs
  --batchsize BATCHSIZE
                        Batch size
  --name NAME           Prefix for save_folders
  --latent_dim LATENT_DIM
                        Latent dimensionality
  --lr_g LR_G           Generator step size
  --beta1_g BETA1_G     Generator Adam beta1
  --beta2_g BETA2_G     Generator Adam beta2
  --lr_d LR_D           Discriminator step size
  --beta1_d BETA1_D     Discriminator Adam beta1
  --beta2_d BETA2_D     Discriminator Adam beta2
  --gp_wt GP_WT         R1 gradient penalty weight
  --ckpt_freq CKPT_FREQ
                        Num of epochs to ckpt after
  --resume_ckpt RESUME_CKPT
                        Resume training at Ckpt #
  --rng RNG             Seed to use for RNGs
  --g_arch G_ARCH       Generator arch fmt "x_y" where x in {"z2", "p4",
                        "p4m"} and y in {"anhir", "lysto", "rotmnist",
                        "cifar10", "food101"}
  --d_arch D_ARCH       Discriminator arch fmt "x_y" where x in {"z2", "p4",
                        "p4m"} and y in {"anhir", "lysto", "rotmnist",
                        "cifar10", "food101"}
  --loss LOSS           GAN loss type. Script currently only supports default.
  --d_updates D_UPDATES
                        Number of D updates per G update
  --dataset DATASET     Dataset to train on. One of {"anhir", "lysto",
                        "rotmnist", "cifar10", "food101"}
```

As we worked only with small datasets that fit on system RAM, the code currently assumes that you have your images `(batch_size, x, y, ch)` and labels `(batch_size, label)` as `train_images.npy` and `train_labels.npy` in a `./data/<dataset_name>/` folder. This will be optimized and generalized for custom datasets soon.

## To-dos
- Add data fetching and pre-processing scripts.
- Add support for tf.data instead of using numpy generators.
- Create an easier way to use custom architectures and datasets.
- Add user-specified weight initializations.
