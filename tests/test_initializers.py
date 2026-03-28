# Code permettant de vérifier que les initialisations des filtres produisent des poids valides

import tensorflow as tf
import numpy as np
from src.models.initializers import get_initializer
from src.analysis.dct import dct2
from src.models.resnet20 import build_resnet20


#Tests pour HeNormalInitializer
#Ces tests vérifient que l'initialiseur HeNormal génère des poids de la bonne forme,
#qu'il est reproductible avec la même graine,
#et qu'il fonctionne correctement avec les couches Conv2D.
def test_get_initializer_he_returns_valid_initializer():
    init = get_initializer("he", seed=42)

    assert init is not None


def test_he_initializer_works_with_conv2d():
    init = get_initializer("he", seed=42)

    layer = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = layer(x)

    assert y.shape == (2, 32, 32, 16)

#Tests pour SigmaInitializer
# Ces tests vérifient que l'initialiseur Sigma génère des poids de la bonne forme, 
# qu'il est reproductible avec la même graine, 
# et qu'il fonctionne correctement avec les couches Conv2D. 
# Un test supplémentaire vérifie que le fallback vers HeNormal fonctionne 
# pour les formes non 3x3.    
def test_get_initializer_sigma_returns_valid_initializer():
    init = get_initializer("sigma", seed=42)
    assert init is not None


def test_sigma_initializer_returns_correct_shape():
    init = get_initializer("sigma", seed=42)
    w = init(shape=(3, 3, 4, 8))

    assert w.shape == (3, 3, 4, 8)


def test_sigma_initializer_works_with_conv2d():
    init = get_initializer("sigma", seed=42)

    layer = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = layer(x)

    assert y.shape == (2, 32, 32, 16)


def test_sigma_initializer_is_reproducible_with_same_seed():
    init1 = get_initializer("sigma", seed=42)
    init2 = get_initializer("sigma", seed=42)

    w1 = init1(shape=(3, 3, 4, 8)).numpy()
    w2 = init2(shape=(3, 3, 4, 8)).numpy()

    assert np.allclose(w1, w2)

# Test pour vérifier que le fallback de SigmaInitializer fonctionne correctement
def test_sigma_initializer_fallback_for_non_3x3_shape():
    init = get_initializer("sigma", seed=42)

    w = init(shape=(1, 1, 8, 16))

    assert w.shape == (1, 1, 8, 16)
    assert np.isfinite(w.numpy()).all()    

#Tests pour GradInitializer
# Ces tests vérifient que l'initialiseur Grad génère des poids de la bonne forme,
# qu'il est reproductible avec la même graine,      
# et qu'il fonctionne correctement avec les couches Conv2D.
# Un test supplémentaire vérifie que le fallback vers HeNormal fonctionne 
# pour les formes non 3x3.
def test_get_initializer_grad_returns_valid_initializer():
    init = get_initializer("grad", seed=42)
    assert init is not None


def test_grad_initializer_returns_correct_shape():
    init = get_initializer("grad", seed=42)
    w = init(shape=(3, 3, 4, 8))

    assert w.shape == (3, 3, 4, 8)


def test_grad_initializer_works_with_conv2d():
    init = get_initializer("grad", seed=42)

    layer = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = layer(x)

    assert y.shape == (2, 32, 32, 16)


def test_grad_initializer_is_reproducible_with_same_seed():
    init1 = get_initializer("grad", seed=42)
    init2 = get_initializer("grad", seed=42)

    w1 = init1(shape=(3, 3, 4, 8)).numpy()
    w2 = init2(shape=(3, 3, 4, 8)).numpy()

    assert np.allclose(w1, w2)


def test_grad_initializer_fallback_for_non_3x3_shape():
    init = get_initializer("grad", seed=42)

    w = init(shape=(1, 1, 8, 16))

    assert w.shape == (1, 1, 8, 16)
    assert np.isfinite(w.numpy()).all()


#Tests pour DCTLowInitializer
# Ces tests vérifient que l'initialiseur DCTLow génère des poids de la bonne forme,
# qu'il est reproductible avec la même graine,  
# et qu'il fonctionne correctement avec les couches Conv2D.
# Un test supplémentaire vérifie que le fallback vers HeNormal fonctionne
# pour les formes non 3x3.  
# Un test final vérifie que les poids générés ont une énergie majoritairement basse fréquence
def test_get_initializer_dctlow_returns_valid_initializer():
    init = get_initializer("dctlow", seed=42)
    assert init is not None


def test_dctlow_initializer_returns_correct_shape():
    init = get_initializer("dctlow", seed=42)
    w = init(shape=(3, 3, 4, 8))

    assert w.shape == (3, 3, 4, 8)


def test_dctlow_initializer_works_with_conv2d():
    init = get_initializer("dctlow", seed=42)

    layer = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = layer(x)

    assert y.shape == (2, 32, 32, 16)


def test_dctlow_initializer_is_reproducible_with_same_seed():
    init1 = get_initializer("dctlow", seed=42)
    init2 = get_initializer("dctlow", seed=42)

    w1 = init1(shape=(3, 3, 4, 8)).numpy()
    w2 = init2(shape=(3, 3, 4, 8)).numpy()

    assert np.allclose(w1, w2)


def test_dctlow_initializer_fallback_for_non_3x3_shape():
    init = get_initializer("dctlow", seed=42)

    w = init(shape=(1, 1, 8, 16))

    assert w.shape == (1, 1, 8, 16)
    assert np.isfinite(w.numpy()).all()


def test_dctlow_initializer_has_more_low_frequency_energy():
    init = get_initializer("dctlow", seed=42)
    w = init(shape=(3, 3, 1, 1)).numpy()[:, :, 0, 0]

    coeffs = dct2(w)
    energy = coeffs ** 2

    low_energy = energy[0, 0] + energy[0, 1] + energy[1, 0] + energy[1, 1]
    total_energy = np.sum(energy)

    assert low_energy / total_energy > 0.8


#Tests pour DCTLowNoiseInitializer
# Ces tests vérifient que l'initialiseur DCTLowNoise génère des poids de la bonne forme,
# qu'il est reproductible avec la même graine,
# et qu'il fonctionne correctement avec les couches Conv2D.
# Un test supplémentaire vérifie que le fallback vers HeNormal fonctionne
# pour les formes non 3x3, et que les poids générés ont une énergie majoritairement basse fréquence.
def test_get_initializer_dctlow_noise_returns_valid_initializer():
    init = get_initializer("dctlow_noise", seed=42)
    assert init is not None


def test_dctlow_noise_initializer_returns_correct_shape():
    init = get_initializer("dctlow_noise", seed=42)
    w = init(shape=(3, 3, 4, 8))

    assert w.shape == (3, 3, 4, 8)


def test_dctlow_noise_initializer_works_with_conv2d():
    init = get_initializer("dctlow_noise", seed=42)

    layer = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = layer(x)

    assert y.shape == (2, 32, 32, 16)


def test_dctlow_noise_initializer_is_reproducible_with_same_seed():
    init1 = get_initializer("dctlow_noise", seed=42)
    init2 = get_initializer("dctlow_noise", seed=42)

    w1 = init1(shape=(3, 3, 4, 8)).numpy()
    w2 = init2(shape=(3, 3, 4, 8)).numpy()

    assert np.allclose(w1, w2)


def test_dctlow_noise_initializer_fallback_for_non_3x3_shape():
    init = get_initializer("dctlow_noise", seed=42)

    w = init(shape=(1, 1, 8, 16))

    assert w.shape == (1, 1, 8, 16)
    assert np.isfinite(w.numpy()).all()


def test_dctlow_noise_initializer_has_majority_low_frequency_energy():
    init = get_initializer("dctlow_noise", seed=42)
    w = init(shape=(3, 3, 1, 1)).numpy()[:, :, 0, 0]

    coeffs = dct2(w)
    energy = coeffs ** 2

    low_energy = energy[0, 0] + energy[0, 1] + energy[1, 0] + energy[1, 1]
    total_energy = np.sum(energy)

    assert low_energy / total_energy > 0.5


def test_dctlow_noise_initializer_differs_from_dctlow():
    init_clean = get_initializer("dctlow", seed=42)
    init_noise = get_initializer("dctlow_noise", seed=42)

    w_clean = init_clean(shape=(3, 3, 1, 1)).numpy()
    w_noise = init_noise(shape=(3, 3, 1, 1)).numpy()

    assert not np.allclose(w_clean, w_noise)


#Tests pour vérifier que les initialisateurs fonctionnent correctement dans le contexte d'un modèle ResNet20
# Ces tests construisent un modèle ResNet20 avec les initialisateurs He et Sigma,
# et vérifient que le modèle peut faire une passe avant avec des données d'entrée aléatoires,
# produisant une sortie de la forme attendue.
def test_build_resnet20_with_he_initializer():
    init = get_initializer("he", seed=42)

    model = build_resnet20(
        input_shape=(32, 32, 3),
        num_classes=10,
        conv_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = model(x)

    assert y.shape == (2, 10)


def test_build_resnet20_with_sigma_initializer():
    init = get_initializer("sigma", seed=42)

    model = build_resnet20(
        input_shape=(32, 32, 3),
        num_classes=10,
        conv_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = model(x)

    assert y.shape == (2, 10)    


#Tests pour vérifier que les initialisateurs fonctionnent correctement dans le contexte d'un modèle ResNet20
# Ces tests construisent un modèle ResNet20 avec les initialisateurs Grad, DCTLow et DCTLowNoise,
# et vérifient que le modèle peut faire une passe avant avec des données d'entrée aléatoires,
#  produisant une sortie de la forme attendue.   
def test_build_resnet20_with_grad_initializer():
    init = get_initializer("grad", seed=42)

    model = build_resnet20(
        input_shape=(32, 32, 3),
        num_classes=10,
        conv_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = model(x)

    assert y.shape == (2, 10)


def test_build_resnet20_with_dctlow_initializer():
    init = get_initializer("dctlow", seed=42)

    model = build_resnet20(
        input_shape=(32, 32, 3),
        num_classes=10,
        conv_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = model(x)

    assert y.shape == (2, 10)


def test_build_resnet20_with_dctlow_noise_initializer():
    init = get_initializer("dctlow_noise", seed=42)

    model = build_resnet20(
        input_shape=(32, 32, 3),
        num_classes=10,
        conv_initializer=init,
    )

    x = tf.random.normal((2, 32, 32, 3))
    y = model(x)

    assert y.shape == (2, 10)    


#Test pour vérifier que les initialisateurs personnalisés sont bien utilisés pour les convolutions 3x3 dans ResNet20
def test_resnet20_uses_custom_initializer_on_3x3_convs():
    init = get_initializer("sigma", seed=42)

    model = build_resnet20(
        input_shape=(32, 32, 3),
        num_classes=10,
        conv_initializer=init,
    )

    conv3x3_layers = []
    conv1x1_layers = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            if layer.kernel_size == (3, 3):
                conv3x3_layers.append(layer)
            elif layer.kernel_size == (1, 1):
                conv1x1_layers.append(layer)

    assert len(conv3x3_layers) > 0
    assert len(conv1x1_layers) > 0

    for layer in conv3x3_layers:
        assert isinstance(layer.kernel_initializer, type(init))    


#Test pour vérifier que les initialisateurs personnalisés ne sont pas utilisés pour les convolutions 1x1 dans ResNet20
# Ce test vérifie que les convolutions 1x1 utilisent l'initialiseur HeNormal, 
# même lorsque l'initialiseur personnalisé est spécifié pour les convolutions 3x3.
def test_resnet20_uses_he_initializer_on_1x1_convs():
    init = get_initializer("sigma", seed=42)

    model = build_resnet20(
        input_shape=(32, 32, 3),
        num_classes=10,
        conv_initializer=init,
    )

    conv1x1_layers = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            if layer.kernel_size == (1, 1):
                conv1x1_layers.append(layer)

    assert len(conv1x1_layers) > 0

    for layer in conv1x1_layers:
        # Vérifie que ce n'est PAS ton initializer custom
        assert not isinstance(layer.kernel_initializer, type(init))        