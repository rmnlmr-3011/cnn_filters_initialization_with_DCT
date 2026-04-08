# Code pour calculer les métriques associées aux filtres, 3 blocs :
	# A. Métriques DCT
	# B. Symétrie et anti-symétrie (beta)
	# C. Agrégations

import numpy as np

from src.analysis.dct import project_filters_to_dct
from src.analysis.dct_bases import LOW_FREQ_SIGMA_GRAD_MASK



# =========================
# Bloc A — métriques DCT
# =========================

# Calcul des coefficients DCT pour tous les noyaux d'une couche CNN
def compute_dct_coefficients(filters_4d: np.ndarray) -> np.ndarray:
    filters_4d = np.asarray(filters_4d, dtype=np.float64)

    if filters_4d.ndim != 4:
        raise ValueError(f"Expected a 4D array, got ndim={filters_4d.ndim}")

    if filters_4d.shape[0] != 3 or filters_4d.shape[1] != 3:
        raise ValueError(
            f"Expected spatial shape (3, 3, in_channels, out_channels), got {filters_4d.shape}"
        )

    return project_filters_to_dct(filters_4d)

# Calcul de l'énergie par composante par noyau
def compute_dct_energy(dct_coeffs: np.ndarray) -> np.ndarray:
    dct_coeffs = np.asarray(dct_coeffs, dtype=np.float64)

    if dct_coeffs.ndim != 4:
        raise ValueError(f"Expected a 4D array, got ndim={dct_coeffs.ndim}")

    if dct_coeffs.shape[0] != 3 or dct_coeffs.shape[1] != 3:
        raise ValueError(
            f"Expected spatial shape (3, 3, in_channels, out_channels), got {dct_coeffs.shape}"
        )

    return np.square(dct_coeffs)


# Calcul de l'énergie totale par noyau
def compute_total_energy(dct_energy: np.ndarray) -> np.ndarray:
    dct_energy = np.asarray(dct_energy, dtype=np.float64)

    if dct_energy.ndim != 4:
        raise ValueError(f"Expected a 4D array, got ndim={dct_energy.ndim}")

    if dct_energy.shape[0] != 3 or dct_energy.shape[1] != 3:
        raise ValueError(
            f"Expected spatial shape (3, 3, in_channels, out_channels), got {dct_energy.shape}"
        )

    return np.sum(dct_energy, axis=(0, 1))

# Calcul du ratio d'énergie basse fréquence / énergie totale par noyau.
# Interprétation : plus ce ratio est élevé, plus le noyau est "simple" (ressemble à une gaussienne ou un edge detector) ; plus il est faible, plus le noyau est "complexe" (ressemble à du bruit haute fréquence)
def compute_low_frequency_ratio(
    dct_energy: np.ndarray,
    mode: str = "sigma_grad"
) -> np.ndarray:
    
    dct_energy = np.asarray(dct_energy, dtype=np.float64)

    if dct_energy.ndim != 4:
        raise ValueError(f"Expected a 4D array, got ndim={dct_energy.ndim}")

    if dct_energy.shape[0] != 3 or dct_energy.shape[1] != 3:
        raise ValueError(
            f"Expected spatial shape (3, 3, in_channels, out_channels), got {dct_energy.shape}"
        )

    if mode != "sigma_grad":
        raise ValueError(
            f"Unsupported mode '{mode}'. Currently supported: 'sigma_grad'."
        )
	
	# Calcul de l'énergie totale
    total_energy = compute_total_energy(dct_energy)

	# Sélection des basses fréquences selon le masque
    mask = LOW_FREQ_SIGMA_GRAD_MASK.astype(np.float64)[:, :, None, None]
    
	# Calcul de l'énergie basse fréquence
    low_freq_energy = np.sum(dct_energy * mask, axis=(0, 1))

	# Calcul du ratio
    ratio = np.divide(
        low_freq_energy,
        total_energy,
        out=np.zeros_like(low_freq_energy, dtype=np.float64),
        where=total_energy > 0,
    )

    return ratio


# =========================
# Bloc B — sym / antisym
# =========================


# La décomposition pair/impair suit cette transformation : f(x,y) ↦ f(−x,−y)

# La formule de beta est la suivante : β^2= (∥fa​∥^2)/(∥fs​∥^2+∥fa​∥^2)​

# Décomposition d'un noyau 3x3 en sa partie symétrique et sa partie anti-symétrique
def decompose_sym_antisym(kernel_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kernel_2d = np.asarray(kernel_2d, dtype=np.float64)

    if kernel_2d.ndim != 2:
        raise ValueError(f"Expected a 2D array, got ndim={kernel_2d.ndim}")

    if kernel_2d.shape != (3, 3):
        raise ValueError(f"Expected shape (3, 3), got {kernel_2d.shape}")

    flipped = np.flip(kernel_2d, axis=(0, 1))  # équivaut à f(-x, -y)

    sym = 0.5 * (kernel_2d + flipped)
    anti = 0.5 * (kernel_2d - flipped)

    return sym, anti

# Calcul du coefficient de symétrie β d'un noyau 3x3
# Interprétation : plus β est proche de 0, plus le noyau est symétrique (ressemble à une gaussienne) ; plus β est proche de 1, plus le noyau est anti-symétrique (ressemble à un edge detector)
def compute_beta_sq(kernel_2d: np.ndarray) -> float:
    sym, anti = decompose_sym_antisym(kernel_2d)

    sym_energy = np.sum(np.square(sym))
    anti_energy = np.sum(np.square(anti))
    total_energy = sym_energy + anti_energy

    if total_energy <= 0:
        return 0.0

    beta_sq = anti_energy / total_energy
    return float(beta_sq)


# =========================
# Bloc C — agrégations
# =========================

# Construit les métriques d'un noyau élémentaire
    # Dictionnaire contenant les infos du noyau ainsi que :
        # - énergie totale
        # - ratio basse fréquence
        # - beta^2
        # - coefficients DCT (flatten)
        # - énergie DCT (flatten)
        
def summarize_kernel_metrics(
    layer_idx: int,
    layer_name: str,
    in_channel: int,
    out_channel: int,
    kernel_2d: np.ndarray,
) -> dict:
    
    kernel_2d = np.asarray(kernel_2d, dtype=np.float64)

    if kernel_2d.ndim != 2:
        raise ValueError(f"Expected a 2D array, got ndim={kernel_2d.ndim}")

    if kernel_2d.shape != (3, 3):
        raise ValueError(f"Expected shape (3, 3), got {kernel_2d.shape}")

    # On passe temporairement par un tenseur 4D minimal : (3, 3, 1, 1)
    filters_4d = kernel_2d[:, :, None, None]

    dct_coeffs = compute_dct_coefficients(filters_4d)
    dct_energy = compute_dct_energy(dct_coeffs)
    total_energy = compute_total_energy(dct_energy)[0, 0]
    low_freq_ratio = compute_low_frequency_ratio(dct_energy)[0, 0]
    beta_sq = compute_beta_sq(kernel_2d)

    coeff_flat = dct_coeffs[:, :, 0, 0].reshape(-1)
    energy_flat = dct_energy[:, :, 0, 0].reshape(-1)

    summary = {
        "layer_idx": int(layer_idx),
        "layer_name": str(layer_name),
        "in_channel": int(in_channel),
        "out_channel": int(out_channel),
        "total_energy": float(total_energy),
        "low_frequency_ratio": float(low_freq_ratio),
        "beta_sq": float(beta_sq),
    }

    # Ajout des 9 coefficients DCT
    for idx, value in enumerate(coeff_flat):
        summary[f"dct_coeff_{idx}"] = float(value)

    # Ajout des 9 énergies DCT
    for idx, value in enumerate(energy_flat):
        summary[f"dct_energy_{idx}"] = float(value)

    return summary


# Agrège les noyaux d'un même filtre (de sortie)

def summarize_filter_metrics(kernel_metrics: list[dict]) -> dict:

    if len(kernel_metrics) == 0:
        raise ValueError("kernel_metrics must not be empty.")

    first = kernel_metrics[0]

    layer_idx = first["layer_idx"]
    layer_name = first["layer_name"]
    out_channel = first["out_channel"]

    # Vérification cohérence
    for km in kernel_metrics:
        if km["layer_idx"] != layer_idx:
            raise ValueError("All kernel metrics must belong to the same layer_idx.")
        if km["layer_name"] != layer_name:
            raise ValueError("All kernel metrics must belong to the same layer_name.")
        if km["out_channel"] != out_channel:
            raise ValueError("All kernel metrics must belong to the same out_channel.")

    total_energy_vals = np.array([km["total_energy"] for km in kernel_metrics], dtype=np.float64)
    low_freq_vals = np.array([km["low_frequency_ratio"] for km in kernel_metrics], dtype=np.float64)
    beta_vals = np.array([km["beta_sq"] for km in kernel_metrics], dtype=np.float64)

    summary = {
        "layer_idx": int(layer_idx),
        "layer_name": str(layer_name),
        "out_channel": int(out_channel),
        "n_kernels": int(len(kernel_metrics)),
        "total_energy_mean": float(np.mean(total_energy_vals)),
        "total_energy_std": float(np.std(total_energy_vals)),
        "low_frequency_ratio_mean": float(np.mean(low_freq_vals)),
        "low_frequency_ratio_std": float(np.std(low_freq_vals)),
        "beta_sq_mean": float(np.mean(beta_vals)),
        "beta_sq_std": float(np.std(beta_vals)),
    }

    # Agrégation moyenne des coefficients DCT et de leur énergie
    coeff_keys = [k for k in first.keys() if k.startswith("dct_coeff_")]
    energy_keys = [k for k in first.keys() if k.startswith("dct_energy_")]

    for key in coeff_keys:
        vals = np.array([km[key] for km in kernel_metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    for key in energy_keys:
        vals = np.array([km[key] for km in kernel_metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    return summary


# Agrège les noyaux d'une même couche
def summarize_layer_metrics(kernel_metrics: list[dict]) -> dict:
    
    if len(kernel_metrics) == 0:
        raise ValueError("kernel_metrics must not be empty.")

    first = kernel_metrics[0]
    layer_idx = first["layer_idx"]
    layer_name = first["layer_name"]

    # Vérification cohérence
    for km in kernel_metrics:
        if km["layer_idx"] != layer_idx:
            raise ValueError("All kernel metrics must belong to the same layer_idx.")
        if km["layer_name"] != layer_name:
            raise ValueError("All kernel metrics must belong to the same layer_name.")

    total_energy_vals = np.array([km["total_energy"] for km in kernel_metrics], dtype=np.float64)
    low_freq_vals = np.array([km["low_frequency_ratio"] for km in kernel_metrics], dtype=np.float64)
    beta_vals = np.array([km["beta_sq"] for km in kernel_metrics], dtype=np.float64)

    summary = {
        "layer_idx": int(layer_idx),
        "layer_name": str(layer_name),
        "n_kernels": int(len(kernel_metrics)),
        "n_out_channels": int(len(set(km["out_channel"] for km in kernel_metrics))),
        "n_in_channels": int(len(set(km["in_channel"] for km in kernel_metrics))),
        "total_energy_mean": float(np.mean(total_energy_vals)),
        "total_energy_std": float(np.std(total_energy_vals)),
        "low_frequency_ratio_mean": float(np.mean(low_freq_vals)),
        "low_frequency_ratio_std": float(np.std(low_freq_vals)),
        "beta_sq_mean": float(np.mean(beta_vals)),
        "beta_sq_std": float(np.std(beta_vals)),
    }

    coeff_keys = [k for k in first.keys() if k.startswith("dct_coeff_")]
    energy_keys = [k for k in first.keys() if k.startswith("dct_energy_")]

    for key in coeff_keys:
        vals = np.array([km[key] for km in kernel_metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    for key in energy_keys:
        vals = np.array([km[key] for km in kernel_metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    return summary
    




##########################################################################

# import numpy as np
# import tensorflow as tf
# from tensorflow.math import multiply, reduce_sum, reduce_mean,reduce_euclidean_norm
# from tensorflow import transpose

# from tensorflow.image import flip_up_down, flip_left_right, rot90

# from cv2 import getDerivKernels



# '''def getDominantAngle(filters):
# 	theta = getSobelTF(filters)
# 	print(filters.shape)
# 	s, a = getSymAntiSymTF(filters)
# 	a_mag = reduce_euclidean_norm(a, axis=[0,1])
# 	s_mag = reduce_euclidean_norm(s, axis=[0,1])

# 	mag = reduce_euclidean_norm(filters, axis=[0,1])


# 	domTheta = []
# 	for i in range(filters.shape[-1]):
# 		x =(a_mag[:,i]*np.cos((theta[:,i]))).numpy()
# 		y =( a_mag[:,i]*np.sin((theta[:,i]))).numpy()


# 		cov = np.cov([x,y])
# 		e_val, e_vec = np.linalg.eig(cov)
# 		e_vec = e_vec[:, np.argmax(e_val)]
# 		e_val = np.max(e_val)
# 		if np.sign(e_vec[0]) != np.sign(x[np.argmax(np.abs(x))]):
# 			e_vec *= -1
# 		domTheta.append(np.arctan2(e_vec[1], e_vec[0]))
# 	#x =a_mag[:,f_num]*np.cos((theta[:,f_num]))
# 	#y = a_mag[:,f_num]*np.sin((theta[:,f_num]))

# 	return np.array(domTheta)'''


# def getDominantAngle(filters):
# 	domTheta = []
# 	vec = []
# 	old_vec = []

# 	_, a = getSymAntiSymTF(filters)
# 	a_mag = reduce_euclidean_norm(a, axis=[0,1])
# 	theta = getSobelTF(filters)
# 	for i in range(filters.shape[-1]):

# 		#print(a_mag.shape)
# 		x =a_mag[:, i]*np.cos((theta[:, i]))
# 		y = a_mag[:, i]*np.sin((theta[:, i]))

# 		u_x = np.mean(x)
# 		#print(u_x)
# 		u_y = np.mean(y)
# 		cov = np.cov([x, y])
# 		e_val, e_vec = np.linalg.eigh(cov)
# 		#print(e_val, e_vec)
# 		e_vec = e_vec[:, np.argmax(e_val)]
# 		e_val = np.max(e_val)

# 		new_vec =    ((e_vec[0] * x + e_vec[1]*y)/(e_vec[0]**2+e_vec[1]**2))[:, None] * e_vec
# 		new_vec = np.mean(new_vec, axis=0)

# 		'''if any((np.sign(new_vec)-np.sign(e_vec))!=0) :
# 			print("DOWN", e_vec, new_vec)
# 		else:
# 			print("OK", e_vec, new_vec)'''
# 		'''if np.sign(e_vec[0]) != np.sign(x[np.argmax(np.abs(x))]):
# 			e_vec *= -1'''
# 			#print(e_val, e_vec)
# 		#print(np.arctan2(e_vec[1], e_vec[0]))
# 		domTheta.append(np.arctan2(new_vec[1], new_vec[0]))
# 		vec.append(new_vec)
# 		#old_vec.append(e_vec)

# 	return np.array(vec), np.array(domTheta)#, old_vec


# def getSobelTF(f):

# 	ksize = f.shape[0]
# 	sobel = getDerivKernels(1,0,ksize=ksize, normalize=True)
# 	sobel_v = -np.expand_dims(np.expand_dims(np.outer(sobel[0], sobel[1]), -1),-1)  # * -1
# 	sobel = getDerivKernels(0,1,ksize=ksize, normalize=True)
# 	sobel_h = np.expand_dims(np.expand_dims(np.outer(sobel[0], sobel[1]), -1),-1)

# 	#print(sobel_h, sobel_v)

# 	s_h = reduce_sum(multiply(f, sobel_h), axis=[0,1])
# 	s_v = reduce_sum(multiply(f, sobel_v), axis=[0,1])


# 	'''sobel_v = np.expand_dims(np.expand_dims(np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=np.float32)/-4., -1),-1)
# 	sobel_h = np.expand_dims(np.expand_dims(np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], dtype=np.float32)/-4., -1) ,-1) 
# 	#print(sobel_h, sobel_v)

# 	s_h = reduce_sum(multiply(f, sobel_h), axis=[0,1])
# 	s_v = reduce_sum(multiply(f, sobel_v), axis=[0,1])'''

# 	return (np.arctan2(s_v,s_h))


# def getSymAntiSymTF(filter):

# 	#patches = extract_image_patches(filters, [1, k, k, 1],  [1, k, k, 1], rates = [1,1,1,1] , padding = 'VALID')
# 	#print(patches)
# 	'''a = filter[0,0,:,:]
# 	b = filter[0,1,:,:]
# 	c = filter[0,2,:,:]
# 	d = filter[1,0,:,:]
# 	e = filter[1,1,:,:]
# 	f = filter[1,2,:,:]
# 	g = filter[2,0,:,:]
# 	h = filter[2,1,:,:]
# 	i = filter[2,2,:,:]

# 	fs1 = expand_dims(a+c+g+i, 0)/4
# 	fs2 = expand_dims(b+d+f+h,0)/4
# 	fs3= expand_dims(e, 0)

# 	sym = stack([concat([fs1, fs2, fs1],  axis=0), 
# 		concat([fs2, fs3, fs2], axis=0),
# 		concat([fs1, fs2, fs1], axis=0)])
		
# 	anti = filter - sym'''

# 	f_reshaped = transpose(filter, perm=[3, 0, 1, 2])
# 	mat_flip_x = flip_left_right(f_reshaped)

# 	mat_flip_y = flip_up_down(f_reshaped)
# 	mat_flip_xy = flip_left_right(flip_up_down(f_reshaped))
# 	#print(mat_flip_x.shape, mat_flip_y.shape, mat_flip_xy.shape)
# 	sum = f_reshaped + mat_flip_x + mat_flip_y + mat_flip_xy
	
# 	mat_sum_rot_90 = rot90(sum)
# 	#gc.collect()
# 	#print("mat_sum_rot_90 shape " , mat_sum_rot_90.shape, self._name)
	
# 	#print("OUT SHAPE," , out.shape)
# 	out = (sum + mat_sum_rot_90) / 8
# 	sym = transpose(out, perm=[1, 2, 3, 0])
# 	anti = filter - sym
# 	return  sym, anti



# Si on réutilise ces fonctions, remplacer get_filter par les fonctions créées dans model_filters.py 


# def topKfilters(model, layer_num, k=10, sev=False):
# 	#print(i, l.name)
# 	filters = get_filter(model, layer_num, sev)

# 	mag = reduce_euclidean_norm(filters, axis=[0,1])**2
# 	avg_mag = reduce_mean(mag, axis=0)
# 	idx = list(range(mag.shape[-1]))
	
# 	idx = [x for _, x in sorted(zip( avg_mag, idx), reverse=True)]
# 	return idx[:int(np.floor(len(idx)*k/100))]


# def topKchannels(model, layer_num, f_num, k=10, sev=False):
# 	#print(i, l.name)
# 	filters = get_filter(model, layer_num, sev)[:,:,:,f_num]

# 	mag = reduce_euclidean_norm(filters, axis=[0,1])**2
# 	#avg_mag = reduce_mean(mag, axis=0)
# 	idx = list(range(mag.shape[-1]))
# 	'''if int((k/100)*len(idx)) == 0:
# 		return idx'''
	
# 	idx = [x for _, x in sorted(zip( mag, idx), reverse=True)]
# 	return idx[:int(np.floor(len(idx)*k/100))]