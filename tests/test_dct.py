# Code permettant de vérifier que la DCT est mathématiquement correcte

import numpy as np
from src.analysis.dct import dct2, idct2

# filtre test (3x3)
kernel = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=np.float32)

# DCT
coeff = dct2(kernel)

# reconstruction
reconstructed = idct2(coeff)

print("Original:")
print(kernel)

print("\nReconstructed:")
print(reconstructed)

print("\nErreur (doit être proche de 0):")
print(np.abs(kernel - reconstructed).mean())