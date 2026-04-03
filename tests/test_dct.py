# Tests permettant de s'assurer que la reconstruction des filtres est bonne, que l'énergue est conservée et qu'il y a une coh.rence dans le batch DCT

import numpy as np

from src.analysis.dct import dct2, idct2, project_filters_to_dct


def test_dct_idct_reconstruction():
    kernel = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    coeffs = dct2(kernel)
    recon = idct2(coeffs)

    assert np.allclose(kernel, recon, atol=1e-10)


def test_parseval_identity():
    kernel = np.array([
        [0.3, -0.1, 0.7],
        [1.2, -0.4, 0.5],
        [-0.8, 0.2, -0.6],
    ], dtype=np.float64)

    coeffs = dct2(kernel)

    energy_spatial = np.sum(kernel ** 2)
    energy_dct = np.sum(coeffs ** 2)

    assert np.allclose(energy_spatial, energy_dct, atol=1e-10)


def test_project_filters_to_dct_shape():
    filters = np.random.randn(3, 3, 4, 6)
    dct_filters = project_filters_to_dct(filters)

    assert dct_filters.shape == (3, 3, 4, 6)


def test_project_filters_to_dct_per_kernel_consistency():
    filters = np.random.randn(3, 3, 2, 3)
    dct_filters = project_filters_to_dct(filters)

    for in_c in range(filters.shape[2]):
        for out_c in range(filters.shape[3]):
            expected = dct2(filters[:, :, in_c, out_c])
            assert np.allclose(dct_filters[:, :, in_c, out_c], expected, atol=1e-10)