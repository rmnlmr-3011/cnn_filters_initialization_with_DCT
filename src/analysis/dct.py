# Code pour appliquer DCT 2D sur les filtres (on passe des poids bruts des filtres à une description en DCT des filtres)

import numpy as np
from scipy.fftpack import dct, idct


# Calcule la transformée DCT d'un noyau 2D
def dct2(kernel_2d: np.ndarray) -> np.ndarray:
    kernel_2d = np.asarray(kernel_2d, dtype=np.float64)
    if kernel_2d.ndim != 2:
        raise ValueError(f"Expected a 2D array, got ndim={kernel_2d.ndim}")
    return dct(dct(kernel_2d.T, norm="ortho").T, norm="ortho")

# Reconstruction du filtre original à partir des coefficients DCT (utile pour debug)
def idct2(coeff_2d: np.ndarray) -> np.ndarray:
    coeff_2d = np.asarray(coeff_2d, dtype=np.float64)
    if coeff_2d.ndim != 2:
        raise ValueError(f"Expected a 2D array, got ndim={coeff_2d.ndim}")
    return idct(idct(coeff_2d.T, norm="ortho").T, norm="ortho")

# Appliquer la transformée DCT à tous les noyaux d'une couche CNN
def project_filters_to_dct(filters_4d: np.ndarray) -> np.ndarray:
    filters_4d = np.asarray(filters_4d, dtype=np.float64)

    if filters_4d.ndim != 4:
        raise ValueError(
            f"Expected a 4D array, got ndim={filters_4d.ndim}"
        )

    if filters_4d.shape[0] != 3 or filters_4d.shape[1] != 3:
        raise ValueError(
            f"Expected spatial shape (3, 3, in_channels, out_channels), got {filters_4d.shape}"
        )

    _, _, in_channels, out_channels = filters_4d.shape
    dct_filters = np.empty_like(filters_4d, dtype=np.float64)

    for in_idx in range(in_channels):
        for out_idx in range(out_channels):
            kernel = filters_4d[:, :, in_idx, out_idx]
            kernel_dct = dct2(kernel)
            dct_filters[:, :, in_idx, out_idx] = kernel_dct

    return dct_filters