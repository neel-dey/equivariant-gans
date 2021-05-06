import argparse


def training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--epochs', type=int, default=100, help='Number of training epochs',
    )
    parser.add_argument(
        '--batchsize', type=int, default=64, help='Batch size',
    )
    parser.add_argument(
        '--name', type=str, default='exp_name', help='Prefix for save_folders',
    )
    parser.add_argument(
        '--latent_dim', type=int, default=128, help='Latent dimensionality',
    )
    parser.add_argument(
        '--lr_g', type=float, default=1e-4, help='Generator step size',
    )
    parser.add_argument(
        '--beta1_g', type=float, default=0.0, help='Generator Adam beta1',
    )
    parser.add_argument(
        '--beta2_g', type=float, default=0.9, help='Generator Adam beta2',
    )
    parser.add_argument(
        '--lr_d', type=float, default=3e-4, help='Discriminator step size',
    )
    parser.add_argument(
        '--beta1_d', type=float, default=0.0, help='Discriminator Adam beta1',
    )
    parser.add_argument(
        '--beta2_d', type=float, default=0.9, help='Discriminator Adam beta2',
    )
    parser.add_argument(
        '--gp_wt', type=float, default=0.1, help='R1 gradient penalty weight',
    )
    parser.add_argument(
        '--ckpt_freq', type=int, default=1, help='Num of epochs to ckpt after',
    )
    parser.add_argument(
        '--resume_ckpt', type=int, default=0, help='Resume training at Ckpt #',
    )
    parser.add_argument(
        '--rng', type=int, default=33, help='Seed to use for RNGs',
    )  # TODO: np.random.seed is controversial now. Better revisit sometime.
    parser.add_argument(
        '--g_arch',
        type=str,
        default='p4_food101',
        help=('Generator arch fmt "x_y" where x in {"z2", "p4", "p4m"} '
              'and y in {"anhir", "lysto", "rotmnist", "cifar10", "food101"}'),
    )
    parser.add_argument(
        '--d_arch',
        type=str,
        default='p4_food101',
        help=('Discriminator arch fmt "x_y" where x in {"z2", "p4", "p4m"} '
              'and y in {"anhir", "lysto", "rotmnist", "cifar10", "food101"}'),
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='relavg_gp',
        help='GAN loss type. Script currently only supports default.',
    )
    parser.add_argument(
        '--d_updates',
        type=int,
        default=2,
        help='Number of D updates per G update',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='food101',
        help=('Dataset to train on. '
              'One of {"anhir", "lysto", "rotmnist", "cifar10", "food101"}'),
    )

    return parser.parse_args()
