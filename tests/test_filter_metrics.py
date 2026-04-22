# Tests permettant de s'assurer quue les métriques des filtres soient bien définies et que le résumé du noyau soit opérationnel

import numpy as np

from src.analysis.filter_metrics import (
    compute_dct_coefficients,
    compute_dct_energy,
    compute_total_energy,
    compute_low_frequency_ratio,
    decompose_sym_antisym,
    compute_beta_sq,
    summarize_kernel_metrics,
)


def test_low_frequency_ratio_dc_kern8el_is_one():
    kernel = np.ones((3, 3), dtype=np.float64)
    filters_4d = kernel[:, :, None, None]

    dct_coeffs = compute_dct_coefficients(filters_4d)
    dct_energy = compute_dct_energy(dct_coeffs)
    ratio = compute_low_frequency_ratio(dct_energy)

    assert ratio.shape == (1, 1)
    assert np.allclose(ratio[0, 0], 1.0, atol=1e-10)


def test_low_frequency_ratio_is_bounded():
    kernel = np.random.randn(3, 3)
    filters_4d = kernel[:, :, None, None]

    dct_coeffs = compute_dct_coefficients(filters_4d)
    dct_energy = compute_dct_energy(dct_coeffs)
    ratio = compute_low_frequency_ratio(dct_energy)[0, 0]

    assert 0.0 <= ratio <= 1.0


def test_decompose_sym_antisym_reconstruction():
    kernel = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    sym, anti = decompose_sym_antisym(kernel)

    assert np.allclose(kernel, sym + anti, atol=1e-10)


def test_beta_sq_symmetric_kernel_is_zero():
    kernel_sym = np.array([
        [1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0],
    ], dtype=np.float64)

    beta_sq = compute_beta_sq(kernel_sym)

    assert np.allclose(beta_sq, 0.0, atol=1e-10)


def test_beta_sq_antisymmetric_kernel_is_one():
    kernel_anti = np.array([
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ], dtype=np.float64)

    beta_sq = compute_beta_sq(kernel_anti)

    assert np.allclose(beta_sq, 1.0, atol=1e-10)


def test_beta_sq_mixed_kernel_is_intermediate():
    kernel_mixed = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)

    beta_sq = compute_beta_sq(kernel_mixed)

    assert 0.0 < beta_sq < 1.0


def test_total_energy_matches_sum_of_dct_energy():
    kernel = np.random.randn(3, 3)
    filters_4d = kernel[:, :, None, None]

    dct_coeffs = compute_dct_coefficients(filters_4d)
    dct_energy = compute_dct_energy(dct_coeffs)
    total_energy = compute_total_energy(dct_energy)

    assert np.allclose(total_energy[0, 0], np.sum(dct_energy[:, :, 0, 0]), atol=1e-10)


def test_summarize_kernel_metrics_contains_expected_keys():
    kernel = np.random.randn(3, 3)

    summary = summarize_kernel_metrics(
        layer_idx=0,
        layer_name="conv2d",
        in_channel=0,
        out_channel=0,
        kernel_2d=kernel,
    )

    expected_keys = {
        "layer_idx",
        "layer_name",
        "in_channel",
        "out_channel",
        "total_energy",
        "low_frequency_ratio",
        "beta_sq",
    }

    assert expected_keys.issubset(summary.keys())

    for i in range(9):
        assert f"dct_coeff_{i}" in summary
        assert f"dct_energy_{i}" in summary