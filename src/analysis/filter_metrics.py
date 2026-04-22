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


def compute_single_dct_energy_ratio(
    dct_energy: np.ndarray,
    row: int,
    col: int,
) -> np.ndarray:
    """
    Calcule, pour chaque noyau, le ratio d'énergie d'une composante DCT donnée :
        dct_energy[row, col] / énergie_totale

    Paramètres
    ----------
    dct_energy:
        Tableau 4D de shape (3, 3, in_channels, out_channels)
    row, col:
        Indices spatiaux de la composante DCT à extraire.

    Retour
    ------
    np.ndarray
        Tableau 2D de shape (in_channels, out_channels)
    """
    dct_energy = np.asarray(dct_energy, dtype=np.float64)

    if dct_energy.ndim != 4:
        raise ValueError(f"Expected a 4D array, got ndim={dct_energy.ndim}")

    if dct_energy.shape[0] != 3 or dct_energy.shape[1] != 3:
        raise ValueError(
            f"Expected spatial shape (3, 3, in_channels, out_channels), got {dct_energy.shape}"
        )

    if not (0 <= row < 3 and 0 <= col < 3):
        raise ValueError(f"Invalid DCT indices ({row}, {col}) for shape (3, 3, ...).")

    total_energy = compute_total_energy(dct_energy)
    component_energy = dct_energy[row, col, :, :]

    ratio = np.divide(
        component_energy,
        total_energy,
        out=np.zeros_like(component_energy, dtype=np.float64),
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
    sigma_energy_ratio = compute_single_dct_energy_ratio(dct_energy, row=0, col=0)[0, 0]
    grad_x_energy_ratio = compute_single_dct_energy_ratio(dct_energy, row=0, col=1)[0, 0]
    grad_y_energy_ratio = compute_single_dct_energy_ratio(dct_energy, row=1, col=0)[0, 0]
    grad_xy_energy_ratio = compute_single_dct_energy_ratio(dct_energy, row=1, col=1)[0, 0]

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
        "sigma_energy_ratio": float(sigma_energy_ratio),
        "grad_x_energy_ratio": float(grad_x_energy_ratio),
        "grad_y_energy_ratio": float(grad_y_energy_ratio),
        "grad_xy_energy_ratio": float(grad_xy_energy_ratio),
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
    sigma_vals = np.array([km["sigma_energy_ratio"] for km in kernel_metrics], dtype=np.float64)
    grad_x_vals = np.array([km["grad_x_energy_ratio"] for km in kernel_metrics], dtype=np.float64)
    grad_y_vals = np.array([km["grad_y_energy_ratio"] for km in kernel_metrics], dtype=np.float64)
    grad_xy_vals = np.array([km["grad_xy_energy_ratio"] for km in kernel_metrics], dtype=np.float64)

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
        "sigma_energy_ratio_mean": float(np.mean(sigma_vals)),
        "sigma_energy_ratio_std": float(np.std(sigma_vals)),
        "grad_x_energy_ratio_mean": float(np.mean(grad_x_vals)),
        "grad_x_energy_ratio_std": float(np.std(grad_x_vals)),
        "grad_y_energy_ratio_mean": float(np.mean(grad_y_vals)),
        "grad_y_energy_ratio_std": float(np.std(grad_y_vals)),
        "grad_xy_energy_ratio_mean": float(np.mean(grad_xy_vals)),
        "grad_xy_energy_ratio_std": float(np.std(grad_xy_vals)),
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
    sigma_vals = np.array([km["sigma_energy_ratio"] for km in kernel_metrics], dtype=np.float64)
    grad_x_vals = np.array([km["grad_x_energy_ratio"] for km in kernel_metrics], dtype=np.float64)
    grad_y_vals = np.array([km["grad_y_energy_ratio"] for km in kernel_metrics], dtype=np.float64)
    grad_xy_vals = np.array([km["grad_xy_energy_ratio"] for km in kernel_metrics], dtype=np.float64)

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
        "sigma_energy_ratio_mean": float(np.mean(sigma_vals)),
        "sigma_energy_ratio_std": float(np.std(sigma_vals)),
        "grad_x_energy_ratio_mean": float(np.mean(grad_x_vals)),
        "grad_x_energy_ratio_std": float(np.std(grad_x_vals)),
        "grad_y_energy_ratio_mean": float(np.mean(grad_y_vals)),
        "grad_y_energy_ratio_std": float(np.std(grad_y_vals)),
        "grad_xy_energy_ratio_mean": float(np.mean(grad_xy_vals)),
        "grad_xy_energy_ratio_std": float(np.std(grad_xy_vals)),
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
    
