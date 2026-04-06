# On implémentera ici les fonctions d'initialisation des filtres

from unicodedata import name

import numpy as np
import tensorflow as tf
from typing import Optional
from src.analysis.dct import idct2



class SigmaPureInitializer(tf.keras.initializers.Initializer):
    """
    Initialisation sigma pure : filtre constant 3x3 sans bruit ni aléatoire.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.float32

        if len(shape) != 4:
            raise ValueError(f"Shape Conv2D attendue (kh, kw, in_ch, out_ch), reçu: {shape}")

        kh, kw, in_channels, out_channels = shape

        # fallback He si pas 3x3
        if kh != 3 or kw != 3:
            he = tf.keras.initializers.HeNormal()
            return he(shape, dtype=dtype)

        # filtre constant
        kernel = np.ones((3, 3), dtype=np.float32)

        # normalisation (important)
        kernel = kernel / np.linalg.norm(kernel)

        kernel = self.scale * kernel

        weights = np.zeros(shape, dtype=np.float32)

        for in_ch in range(in_channels):
            for out_ch in range(out_channels):
                weights[:, :, in_ch, out_ch] = kernel

        return tf.convert_to_tensor(weights, dtype=dtype)

    def get_config(self):
        return {
            "scale": self.scale,
        }
    

class SigmaInitializer(tf.keras.initializers.Initializer):
    """
    Initialisation custom qui génère des filtres 3x3 symétriques et lisses.
    Pour les autres tailles de noyaux, on retombe sur HeNormal.
    """

    def __init__(self, seed: Optional[int] = None, scale: float = 1.0):
        self.seed = seed
        self.scale = scale

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.float32

        # shape attendue pour Conv2D : (kernel_h, kernel_w, in_channels, out_channels)
        if len(shape) != 4:
            raise ValueError(f"Shape Conv2D attendue (kh, kw, in_ch, out_ch), reçu: {shape}")

        kh, kw, in_channels, out_channels = shape

        # Si ce n'est pas un filtre 3x3, on utilise HeNormal comme fallback
        if kh != 3 or kw != 3:
            he = tf.keras.initializers.HeNormal(seed=self.seed)
            return he(shape, dtype=dtype)

        rng = np.random.default_rng(self.seed)

        weights = np.zeros(shape, dtype=np.float32)

        for in_ch in range(in_channels):
            for out_ch in range(out_channels):
                # On construit un noyau 3x3 symétrique autour du centre
                center = rng.normal(loc=0.0, scale=1.0)
                edge = rng.normal(loc=0.0, scale=0.5)
                corner = rng.normal(loc=0.0, scale=0.25)

                kernel = np.array(
                    [
                        [corner, edge, corner],
                        [edge, center, edge],
                        [corner, edge, corner],
                    ],
                    dtype=np.float32,
                )

                # Normalisation simple pour éviter des valeurs trop grandes
                norm = np.linalg.norm(kernel)
                if norm > 0:
                    kernel = kernel / norm

                kernel = self.scale * kernel
                weights[:, :, in_ch, out_ch] = kernel

        return tf.convert_to_tensor(weights, dtype=dtype)

    def get_config(self):
        return {
            "seed": self.seed,
            "scale": self.scale,
        }


class GradVerticalPureInitializer(tf.keras.initializers.Initializer):
    """
    Initialisation déterministe avec un filtre de gradient vertical pur.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.float32

        if len(shape) != 4:
            raise ValueError(f"Shape Conv2D attendue (kh, kw, in_ch, out_ch), reçu: {shape}")

        kh, kw, in_channels, out_channels = shape

        if kh != 3 or kw != 3:
            he = tf.keras.initializers.HeNormal()
            return he(shape, dtype=dtype)

        kernel = np.array(
            [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1],
            ],
            dtype=np.float32,
        )

        kernel = kernel / np.linalg.norm(kernel)
        kernel = self.scale * kernel

        weights = np.zeros(shape, dtype=np.float32)

        for in_ch in range(in_channels):
            for out_ch in range(out_channels):
                weights[:, :, in_ch, out_ch] = kernel

        return tf.convert_to_tensor(weights, dtype=dtype)

    def get_config(self):
        return {
            "scale": self.scale,
        }


class GradHorizontalPureInitializer(tf.keras.initializers.Initializer):
    """
    Initialisation déterministe avec un filtre de gradient horizontal pur.
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.float32

        if len(shape) != 4:
            raise ValueError(f"Shape Conv2D attendue (kh, kw, in_ch, out_ch), reçu: {shape}")

        kh, kw, in_channels, out_channels = shape

        if kh != 3 or kw != 3:
            he = tf.keras.initializers.HeNormal()
            return he(shape, dtype=dtype)

        kernel = np.array(
            [
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )

        kernel = kernel / np.linalg.norm(kernel)
        kernel = self.scale * kernel

        weights = np.zeros(shape, dtype=np.float32)

        for in_ch in range(in_channels):
            for out_ch in range(out_channels):
                weights[:, :, in_ch, out_ch] = kernel

        return tf.convert_to_tensor(weights, dtype=dtype)

    def get_config(self):
        return {
            "scale": self.scale,
        }


class GradInitializer(tf.keras.initializers.Initializer):
    """
    Initialisation custom basée sur des motifs de gradient 3x3.
    Pour les autres tailles de noyaux, on retombe sur HeNormal.
    """

    def __init__(self, seed: Optional[int] = None, scale: float = 1.0):
        self.seed = seed
        self.scale = scale

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.float32

        if len(shape) != 4:
            raise ValueError(f"Shape Conv2D attendue (kh, kw, in_ch, out_ch), reçu: {shape}")

        kh, kw, in_channels, out_channels = shape

        if kh != 3 or kw != 3:
            he = tf.keras.initializers.HeNormal(seed=self.seed)
            return he(shape, dtype=dtype)

        rng = np.random.default_rng(self.seed)
        weights = np.zeros(shape, dtype=np.float32)

        # Bases de gradients 3x3
        grad_bases = [
            #Gradient Vertical
            np.array(
                [
                    [-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1],
                ],
                dtype=np.float32,
            ),
            #Gradient Horizontal
            np.array(
                [
                    [-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1],
                ],
                dtype=np.float32,
            ),
            #Gradient Diagonal 1
            np.array(
                [
                    [0, 1, 1],
                    [-1, 0, 1],
                    [-1, -1, 0],
                ],
                dtype=np.float32,
            ),
            #Gradient Diagonal 2
            np.array(
                [
                    [1, 1, 0],
                    [1, 0, -1],
                    [0, -1, -1],
                ],
                dtype=np.float32,
            ),
        ]

        for in_ch in range(in_channels):
            for out_ch in range(out_channels):
                base = grad_bases[rng.integers(0, len(grad_bases))].copy()

                # Petit bruit pour éviter que tous les filtres soient identiques
                noise = rng.normal(loc=0.0, scale=0.05, size=(3, 3)).astype(np.float32)
                kernel = base + noise
                # Normalisation simple pour éviter des valeurs trop grandes
                norm = np.linalg.norm(kernel)
                if norm > 0:
                    kernel = kernel / norm

                kernel = self.scale * kernel
                weights[:, :, in_ch, out_ch] = kernel

        return tf.convert_to_tensor(weights, dtype=dtype)

    def get_config(self):
        return {
            "seed": self.seed,
            "scale": self.scale,
        }
    

class DCTLowInitializer(tf.keras.initializers.Initializer):
    """
    Initialisation custom qui favorise les basses fréquences en espace DCT.
    Pour les autres tailles de noyaux, on retombe sur HeNormal.
    """

    def __init__(self, seed: Optional[int] = None, scale: float = 1.0):
        self.seed = seed
        self.scale = scale

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.float32

        if len(shape) != 4:
            raise ValueError(f"Shape Conv2D attendue (kh, kw, in_ch, out_ch), reçu: {shape}")

        kh, kw, in_channels, out_channels = shape

        if kh != 3 or kw != 3:
            he = tf.keras.initializers.HeNormal(seed=self.seed)
            return he(shape, dtype=dtype)

        rng = np.random.default_rng(self.seed)
        weights = np.zeros(shape, dtype=np.float32)

        low_freq_mask = np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )

        for in_ch in range(in_channels):
            for out_ch in range(out_channels):
                dct_coeffs = rng.normal(loc=0.0, scale=1.0, size=(3, 3)).astype(np.float32)
                dct_coeffs = dct_coeffs * low_freq_mask

                kernel = idct2(dct_coeffs).astype(np.float32)

                norm = np.linalg.norm(kernel)
                if norm > 0:
                    kernel = kernel / norm

                kernel = self.scale * kernel
                weights[:, :, in_ch, out_ch] = kernel

        return tf.convert_to_tensor(weights, dtype=dtype)

    def get_config(self):
        return {
            "seed": self.seed,
            "scale": self.scale,
        }


class DCTLowNoiseInitializer(tf.keras.initializers.Initializer):
    """
    Initialisation custom basée sur DCT-low avec ajout d'un petit bruit spatial.
    Pour les autres tailles de noyaux, on retombe sur HeNormal.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        scale: float = 1.0,
        noise_std: float = 0.05,
    ):
        self.seed = seed
        self.scale = scale
        self.noise_std = noise_std

    def __call__(self, shape, dtype=None):
        dtype = dtype or tf.float32

        if len(shape) != 4:
            raise ValueError(f"Shape Conv2D attendue (kh, kw, in_ch, out_ch), reçu: {shape}")

        kh, kw, in_channels, out_channels = shape

        if kh != 3 or kw != 3:
            he = tf.keras.initializers.HeNormal(seed=self.seed)
            return he(shape, dtype=dtype)

        rng = np.random.default_rng(self.seed)
        weights = np.zeros(shape, dtype=np.float32)

        low_freq_mask = np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )

        for in_ch in range(in_channels):
            for out_ch in range(out_channels):
                dct_coeffs = rng.normal(loc=0.0, scale=1.0, size=(3, 3)).astype(np.float32)
                dct_coeffs = dct_coeffs * low_freq_mask

                kernel = idct2(dct_coeffs).astype(np.float32)

                noise = rng.normal(loc=0.0, scale=self.noise_std, size=(3, 3)).astype(np.float32)
                kernel = kernel + noise

                norm = np.linalg.norm(kernel)
                if norm > 0:
                    kernel = kernel / norm

                kernel = self.scale * kernel
                weights[:, :, in_ch, out_ch] = kernel

        return tf.convert_to_tensor(weights, dtype=dtype)

    def get_config(self):
        return {
            "seed": self.seed,
            "scale": self.scale,
            "noise_std": self.noise_std,
        }


def get_initializer(name: str, seed: Optional[int] = None, **kwargs):
    """
    Retourne un initialiseur Keras selon son nom.
    """
    name = name.lower()

    if name == "he":
        return tf.keras.initializers.HeNormal(seed=seed)

    if name == "sigma_pure":
        return SigmaPureInitializer(**kwargs)

    if name == "sigma":
        return SigmaInitializer(seed=seed, **kwargs)
    
    if name == "grad_vertical_pure":
        return GradVerticalPureInitializer(**kwargs)

    if name == "grad_horizontal_pure":
        return GradHorizontalPureInitializer(**kwargs)

    if name == "grad":
        return GradInitializer(seed=seed, **kwargs)
    
    if name == "dctlow":
        return DCTLowInitializer(seed=seed, **kwargs)

    if name == "dctlow_noise":
        return DCTLowNoiseInitializer(seed=seed, **kwargs)
        
    raise ValueError(f"Initialiseur inconnu: {name}")
