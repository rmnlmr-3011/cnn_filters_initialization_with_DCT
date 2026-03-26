# Code pour appliquer DCT 2D sur les filtres (on passe des poids bruts des filtres à une description en DCT des filtres)

import numpy as np

from scipy.fftpack import dct, idct


def dct2(a):
	return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
	return idct(idct(a.T, norm='ortho').T, norm='ortho')    