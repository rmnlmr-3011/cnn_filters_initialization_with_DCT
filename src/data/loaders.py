# On chargera ici les datasets
import numpy as np
from tensorflow.keras.datasets import cifar10


def load_cifar10(normalize=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Flatten labels
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Normalisation
    if normalize:
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


def split_train_val(x_train, y_train, val_ratio=0.1, seed=42, shuffle=True):
    """
    Sépare le jeu d'entraînement en sous-ensembles train et validation.

    Parameters
    ----------
    x_train : np.ndarray
        Images d'entraînement.
    y_train : np.ndarray
        Labels d'entraînement.
    val_ratio : float, default=0.1
        Proportion du jeu d'entraînement à utiliser pour la validation.
    seed : int, default=42
        Seed pour garantir la reproductibilité du split.
    shuffle : bool, default=True
        Mélange les données avant découpage.

    Returns
    -------
    (x_train_new, y_train_new), (x_val, y_val)
    """
    n_samples = len(x_train)
    n_val = int(n_samples * val_ratio)

    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    x_train = x_train[indices]
    y_train = y_train[indices]

    x_val = x_train[:n_val]
    y_val = y_train[:n_val]

    x_train_new = x_train[n_val:]
    y_train_new = y_train[n_val:]

    return (x_train_new, y_train_new), (x_val, y_val)
