import math
from typing import List
from warnings import filterwarnings

import numpy as np
from tqdm import tqdm

filterwarnings("ignore")

def partition_box(data: np.ndarray, boxsize: float, gridsize: float) -> List[float]:
    """Sorts all data points into a 3D grid with `cells per side = boxsize / gridsize`

    Parameters
    ----------
    data : np.ndarray
        `(N, d)` array with all data points' coordinates, where `N` is the
        number of data points and `d` the dimensions
    boxsize : float
        Simulation box size (per side)
    gridsize : float
        Grid size (per side)

    Returns
    -------
    List[float]

    """
    # Number of grid cells per side.
    cells_per_side = int(math.ceil(boxsize / gridsize))
    # Grid ID for each data point.
    grid_id = (data / gridsize).astype(int, copy=False)
    # Correction for points on the edges.
    grid_id[np.where(grid_id == cells_per_side)] = cells_per_side - 1

    # This list stores all of the particles original IDs in a convenient 3D
    # list. It is kind of a pointer of size n_cpd**3
    data_cell_id = [[] for _ in range(cells_per_side**3)]
    cells = (
        cells_per_side**2 * grid_id[:, 0]
        + cells_per_side * grid_id[:, 1]
        + grid_id[:, 2]
    )
    for cell in tqdm(
        range(np.size(data, 0)), desc="Partitioning box", ncols=100, colour="blue"
    ):
        data_cell_id[cells[cell]].append(cell)

    return data_cell_id


if __name__ == "__main__":
    pass
