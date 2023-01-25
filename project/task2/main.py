import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

sys.path.append('..')  # To allow imports from Task 1 folder
from task1.main import D, brownian_N_2D

DX_MM = 0.004
DT_S = 0.01

@njit
def dx_effective(x, y, tumor_pos, m, area, tk):
    """
    Calculate the effective dx value for points in the xy-plane.

    Parameters
    ----------
    x : np.ndarray
        (nx,) array with positions in x direction.
    y : np.ndarray
        (ny,) array with positions in y direction.
    tumor_pos : np.ndarray
        ( m, (x,y) ) array with positions of centers of tumors.
    m : int
        Number of tumors.
    area : float
        Area of tumors. Same length unit as position.
    tk : np.ndarray
        (m,) array with tumor coefficients.

    Returns
    -------
    np.ndarray
        (x, y) array with effective dx value
    """
    nx, ny = len(x), len(y)
    tk_eff = np.ones((nx, ny))
    radius = np.sqrt(area / np.pi)

    for xi, x_val in enumerate(x):
        for yi, y_val in enumerate(y):
            for mi in range(m):
                tumor = tumor_pos[mi]
                x_dist = tumor[0] - x_val
                y_dist = tumor[1] - y_val
                pos = np.asarray((x_dist,y_dist))
                dist = np.linalg.norm(pos)
                if dist < radius:
                    tk_eff[xi,yi] *= tk[mi]

    return DX_MM * np.sqrt(np.transpose(tk_eff))

def task_2a():
    print(f'D = {D(0.004,0.01):.1e}')

def task_2c():
    # Valus given in task
    m = 15
    L = 0.02  # [mm]
    nx, ny = 500, 500

    tumor_pos = np.random.uniform(0, L, size=(m,2))
    area = np.pi * DX_MM ** 2
    tk = np.full(shape=m, fill_value=0.1)
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)

    # Simulation
    dx_eff = dx_effective(x, y, tumor_pos, m, area, tk)
    c = plt.pcolormesh(x, y, dx_eff, cmap='plasma_r')
    plt.colorbar(c)
    plt.show()

def main():
    # task_2a()
    task_2c()

if __name__ == '__main__':
    main()
