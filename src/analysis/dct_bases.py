# Fichier comportant les définitions des bases DCT 3x3 et des masques de sélection pour les composantes DCT

import numpy as np

from src.analysis.dct import idct2


# =========================
# Indices nommés des composantes DCT 3x3
# =========================

SIGMA_IDX = (0, 0)
GRAD_X_IDX = (0, 1)
GRAD_Y_IDX = (1, 0)


# =========================
# Masques de sélection
# =========================

LOW_FREQ_SIGMA_GRAD_MASK = np.array([
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
], dtype=bool)

DC_ONLY_MASK = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=bool)

GRAD_X_ONLY_MASK = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
], dtype=bool)

GRAD_Y_ONLY_MASK = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 0, 0],
], dtype=bool)

GRAD_ONLY_MASK = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
], dtype=bool)


# =========================
# Base DCT 3x3
# =========================

# Fonction pour générer les bases DCT 3x3
# Permettra de visualiser les composantes DCT
def get_dct_basis_3x3() -> np.ndarray:

    basis = np.zeros((3, 3, 3, 3), dtype=np.float64)

    for u in range(3):
        for v in range(3):
            coeff = np.zeros((3, 3), dtype=np.float64)
            coeff[u, v] = 1.0
            basis[u, v] = idct2(coeff)

    return basis