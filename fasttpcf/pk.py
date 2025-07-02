import math
from typing import Tuple

import MAS_library as MASL
import numpy as np
import Pk_library as PKL

__all__ = ['cross_power_spectrum']


def fold_box(
    pos: np.ndarray, 
    boxsize: float,
) -> np.ndarray: 
    """Box folding. See equation 58 in Colombi et al. 2009 
    (DOI: 10.1111/j.1365-2966.2008.14176.x)

    Parameters
    ----------
    pos : np.ndarray
        The array of X/Y/Z positions for the set points.
    boxsize : float
        Size of simulation box.

    Returns
    -------
    np.ndarray
        New X/Y/Z positions for the set of points.
    """
    folded_box = np.zeros_like(pos)
    for i in range(3):
        mask = pos[:, i] <= np.ceil(boxsize / 2)
        folded_box[mask, i] = 2.0 * pos[mask, i]
        folded_box[~mask, i] = 2.0 * pos[~mask, i] - boxsize
    return folded_box


def measure_pk(
    data_1: np.ndarray,
    data_2: np.ndarray,
    boxsize: float,
    fold_factor: int = 0,
    n_grid: int = 512,
    nthreads: int = 16,
) -> Tuple[np.ndarray]:
    """Measure the cross-power spectrum.

    Parameters
    ----------
    data_1 : np.ndarray
        The array of X/Y/Z positions for the first set points.
    data_2 : np.ndarray
        The array of X/Y/Z positions for the second set points.
    boxsize : float
        Size of simulation box.
    n_grid : int, optional
        Fourier modes grid, by default 512

    Returns
    -------
    Tuple[np.ndarray]
        Fourier modes and cross-power spectrum
    """

    # Compute CIC correlation for data_1
    delta1 = np.zeros((n_grid, n_grid, n_grid), dtype=np.float32)
    MASL.MA(data_1, delta1, boxsize, MAS="CIC", verbose=False)
    delta1 /= np.mean(delta1, dtype=np.float32)
    delta1 -= 1.0
    
    # Compute CIC correlation for data_2
    delta2 = np.zeros((n_grid, n_grid, n_grid), dtype=np.float32)
    MASL.MA(data_2, delta2, boxsize, MAS="CIC", verbose=False)
    delta2 /= np.mean(delta2, dtype=np.float64)
    delta2 -= 1.0

    # Measure total power spectrum
    Pkdd = PKL.XPk([delta1, delta2], boxsize, axis=0, MAS=["CIC", "CIC"], 
                   threads=nthreads)
    # Pk0_X[i, :]  = Pkhm.XPk[:,0,0] #monopole of 1-2 cross P(k)
    pk = Pkdd.XPk[:, 0, 0]
    k = Pkdd.k3D
    if fold_factor > 0:
        k *= 2**fold_factor
    
    return (k, pk)


def cross_power_spectrum(
    data_1: np.ndarray,
    data_2: np.ndarray,
    boxsize: float,
    k_max: float,
    n_grid: int = 512,
    nthreads: int = 16,
):
    # Nyquist frequency
    k_ny = np.pi * float(n_grid / boxsize)

    # If no foldings are required, compute cross-power spectrum and exit
    n_folds = math.ceil(k_max / k_ny)

    if k_max <= k_ny:
        print(f"No foldings required.")
        k, pk = measure_pk(
            data_1=data_1,
            data_2=data_2, 
            boxsize=boxsize,
            fold_factor=0,
            n_grid=n_grid,
            nthreads=nthreads
        )
        return k, pk
    else:
        print(f"Performing {n_folds} folds to reach desired k_max.")

    # Compute cross-power spectrum for each folding and aggregate results.
    k_all, pk_all = [], []
    for i in range(n_folds + 1):
        if i > 0:
            data_1 = fold_box(data_1, boxsize)
            data_2 = fold_box(data_2, boxsize)
        k, pk = measure_pk(
            data_1=data_1,
            data_2=data_2, 
            boxsize=boxsize,
            fold_factor=i,
            n_grid=n_grid,
            nthreads=nthreads
        )
        k_all.append(k)
        pk_all.append(pk)

    # Each folding's new Nyquist frequency
    k_nys = [2**i * k_ny for i in range(n_folds + 1)]

    mask_first = (k_all[0] < 0.5 * k_nys[0])
    k_first = k_all[0][mask_first]
    pk_first = pk_all[0][mask_first]
    
    mask_last = (k_all[-1] > 0.5 * k_nys[-1])
    k_last = k_all[-1][mask_last]
    pk_last = pk_all[-1][mask_last]

    k_foldings = []
    pk_foldings = []
    if n_folds > 1:
        for i in range(1, n_folds):
            mask = (k_all[i] > 0.5 * k_nys[i-1]) & (k_all[i] < 0.5 * k_nys[i])
            k_foldings.append(k_all[i][mask])
            pk_foldings.append(pk_all[i][mask])
        k_foldings = np.concatenate(k_foldings)
        pk_foldings = np.concatenate(pk_foldings)

    # Concatenate results.
    k_total = np.hstack([k_first, k_foldings, k_last])
    pk_total = np.hstack([pk_first, pk_foldings, pk_last])

    return k_total, pk_total


if __name__ == "__main__":
    pass
