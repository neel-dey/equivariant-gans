import numpy as np
from tensorflow.keras.utils import to_categorical


def npy_loader(dataset, num_classes):
    """
    Utility function to load npy files corresponding to training images and
    labels.
    Args:
        dataset: str
            Name of dataset. One of {'anhir5', 'lysto', cifar10', 'food101',
            'rotmnist'}.
        num_classes: int
            Number of categories in the image set.
    """

    data = np.load('./data/{}/train_images.npy'.format(dataset))
    labels = np.load('./data/{}/train_labels.npy'.format(dataset))
    labels = to_categorical(labels, num_classes=num_classes)

    return data, labels


def dataset_lookup(dataset):
    """
    Utility function to return the number of classes for a chosen dataset.
    Args:
        dataset: str
            Name of dataset. One of {'anhir5', 'lysto', cifar10', 'food101',
            'rotmnist'}
    """
    if dataset == 'anhir5':
        num_classes = 5
        print_multiplier = 8
    elif dataset == 'lysto':
        num_classes = 3
        print_multiplier = 8
    elif dataset == 'cifar10':
        num_classes = 10
        print_multiplier = 1
    elif dataset == 'rotmnist':
        num_classes = 10
        print_multiplier = 1
    elif dataset == 'food101':
        num_classes = 101
        print_multiplier = 1
    else:
        raise ValueError
    return num_classes, print_multiplier


def data_normalizer(tensor, dataset):
    """
    Rescale intensities of the input dataset.
    Args:
        tensor: np array
            Batch to rescale.
        dataset: str
            Name of dataset. One of {'anhir5', 'lysto', cifar10', 'food101',
            'rotmnist'}
    """
    if dataset == 'rotmnist':
        return (tensor * 2.0) - 1.0
    else:
        return (tensor/127.5) - 1.0


def data_generator(data, labels, batch_size, noise_dim, dataset):
    """
    Numpy data generator for GAN training.
    Args:
        data: np array
            Overall image data array to sample from.
        labels: np array
            Overall label array to sample from.
        batch_size: int
            Batch size to return.
        noise_dim:
            Latent dimensionality.
        dataset: str
            Name of dataset. One of {'anhir5', 'lysto', cifar10', 'food101',
            'rotmnist'}
    """
    datasize = data.shape[0]
    while True:
        image_idx = np.random.randint(0, datasize, batch_size)

        noise = np.random.randn(batch_size, noise_dim).astype(np.float32)
        real_images_batch = data_normalizer(data[image_idx], dataset)
        real_labels_batch = labels[image_idx]

        yield noise, real_images_batch, real_labels_batch


def generator_dimensionality(gen_arch):
    """
    Return dimensionalities used in paper corresponding to the datasets chosen.

    Args:
        gen_arch: str
            Generator arch fmt "x_y" where x in {"z2", "p4", "p4m"}
            and y in {"anhir", "lysto", "rotmnist", "cifar10", "food101"}
    """
    # To keep the number of parameters across settings roughly consistent:
    if 'p4_' in gen_arch:
        channel_scale_factor = np.sqrt(4)
    if 'p4m_' in gen_arch:
        channel_scale_factor = np.sqrt(8)
    if 'z2_' in gen_arch:
        channel_scale_factor = 1

    if '_anhir5' in gen_arch:
        projection_dimensionality = 128 * 4 * 4  # latent linear projection
        projection_reshape = (4, 4, 128)
        label_emb_dim = 128  # Embedding dimension for one-hot label vector.
    elif '_lysto' in gen_arch:
        projection_dimensionality = 128 * 4 * 4
        projection_reshape = (4, 4, 128)
        label_emb_dim = 128
    elif '_cifar10' in gen_arch:
        projection_dimensionality = 256 * 4 * 4
        projection_reshape = (4, 4, 256)
        label_emb_dim = 128
    elif '_rotmnist' in gen_arch:
        projection_dimensionality = 128 * 7 * 7
        projection_reshape = (7, 7, 128)
        label_emb_dim = 64
    elif '_food101' in gen_arch:
        projection_dimensionality = 1024 * 4 * 4
        projection_reshape = (4, 4, 1024)
        label_emb_dim = 64
    else:
        raise ValueError

    return (
        channel_scale_factor,
        projection_dimensionality,
        projection_reshape,
        label_emb_dim
    )
