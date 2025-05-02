import math
from typing import Tuple

import numpy as np
from tqdm import tqdm

from fasttpcf.partition import partition_box

__all__ = ['cross_tpcf_jk_radial']


def density(
    n_obj: int,
    radii: np.ndarray,
    radial_edges: np.ndarray,
    mass: float,
) -> np.ndarray:
    """Compute density profile as a function of radial separation r.

    Parameters
    ----------
    n_obj : int
        Number of haloes.
    radii : np.ndarray
        Radial coordinate for all particles.
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    mass : float
        Particle mass.

    Returns
    -------
    np.ndarray
        Density profile.
    """
    n_bins = radial_edges.shape[0] - 1
    volume_shell = 4.0 / 3.0 * np.pi * np.diff((np.power(radial_edges, 3)))

    # Compute mass density per spherical shell
    rho = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (radial_edges[i] < radii) & (radii <= radial_edges[i + 1])
        rho[i] = mass * mask.sum() / (n_obj * volume_shell[i])
    return rho


def density_jk(
    n_obj_d1: int,
    data_1_id: list,
    data_1_hid: np.ndarray,
    radial_data: np.ndarray,
    radial_edges: np.ndarray,
    radial_data_1_id: np.ndarray,
    boxsize: float,
    gridsize: float,
    mass: float,
) -> Tuple[np.ndarray]:
    """Density profile jackknife samples

    Parameters
    ----------
    n_obj_d1 : int
        Number of haloes.
    data_1_id : list
        Box partitioning 3D grid.
    data_1_hid : np.ndarray
        Halo ID.
    radial_data : np.ndarray
        r coordinate for all particles
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    radial_data_1_id : np.ndarray
        Parent halo ID for each particle.
    boxsize : float
        Size of simulation box
    gridsize : float
        Size of sub-volume or cell of the box
    mass : float
        Particle mass.

    Returns
    -------
    Tuple[np.ndarray]
        Returns a tuple with `(rho, rho_samples, rho_mean, cov)`, where `rho` is
        the total correlation function measured directly on the full simulation
        box. `rho_samples` is an array of shape `(Njk, Nbins)` with the Njk
        samples of the density profile. `rho_mean` and `rho_cov` are the mean
        and covariance of `rho_samples`.
    """
    n_bins = radial_edges.shape[0] - 1
    # Number of cells per dimension
    cells_per_side = int(math.ceil(boxsize / gridsize))

    # Number of jackknife samples. One sample per cell
    n_jk_samples = cells_per_side**3

    # Data 1 index array
    data_1_row_idx = np.arange(n_obj_d1)

    rho_samples = np.zeros((n_jk_samples, n_bins))
    for sample in tqdm(
        range(n_jk_samples), desc="Pair counting", ncols=100, colour="green"
    ):
        d1_total_sample = np.size(data_1_row_idx[data_1_id[sample]], 0)
        mask = np.isin(radial_data_1_id, data_1_hid[data_1_id[sample]])

        rho_samples[sample] = density(
            n_obj=d1_total_sample,
            radii=radial_data[mask],
            radial_edges=radial_edges,
            mass=mass,
        )

    # Compute mean correlation function from all jk samples
    rho_mean = np.mean(rho_samples, axis=0)

    # Compute covariance matrix of the radial bins using all jk samples
    rho_cov = (
        (float(n_jk_samples) - 1.0)
        * np.cov(rho_samples.T, bias=True)
        / np.sqrt(n_obj_d1)
    )

    rho = density(
        n_obj=n_obj_d1, radii=radial_data, radial_edges=radial_edges, mass=mass
    )

    return rho, rho_samples, rho_mean, rho_cov


def cross_tpcf_jk_radial(
    data_1: np.ndarray,
    data_1_hid: np.ndarray,
    radial_data: np.ndarray,
    radial_edges: np.ndarray,
    radial_data_hid: np.ndarray,
    boxsize: float,
    gridsize: float,
    mass: float,
    jk_estimates: bool = True,
) -> Tuple[np.ndarray]:
    """Compute the cross-correlation function between data 1 and data 2. It is
    assumed that data 1...

    Parameters
    ----------
    data_1 : np.ndarray
        The array of X/Y/Z positions for the first set of points. Calculations
        are done in the precision of the supplied arrays.
    data_1_hid : np.ndarray
        Halo ID.
    radial_data : np.ndarray
        r coordinate for all particles
    radial_edges : np.ndarray
        The bins need to be contiguous and sorted in increasing order (smallest
        bins come first).
    radial_data_hid : np.ndarray
        Parent halo ID for each particle.
    boxsize : float
        Size of simulation box
    gridsize : float
        Size of sub-volume or cell of the box
    mass : float
        Particle mass.
    jk_estimates : bool, optional
        If True returns all the jackknife samples and their mean, by default True

    Returns
    -------
    Tuple[np.ndarray]
        Total density profile and covariance matrix. If `jk_estimates` is
        True, it also returns the jackknife samples and their mean.
    """
    # Partition box
    data_1_id = partition_box(
        data=data_1,
        boxsize=boxsize,
        gridsize=gridsize,
    )

    rho, rho_samples, rho_mean, rho_cov = density_jk(
        n_obj_d1=np.size(data_1, 0),
        data_1_id=data_1_id,
        data_1_hid=data_1_hid,
        radial_data=radial_data,
        radial_edges=radial_edges,
        radial_data_1_id=radial_data_hid,
        boxsize=boxsize,
        gridsize=gridsize,
        mass=mass,
    )
    if jk_estimates:
        return rho, rho_samples, rho_mean, rho_cov
    else:
        return rho, rho_cov


if __name__ == "__main__":
    pass
