import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

sys.path.append('..')  # To allow imports from Task 1 folder
from task1.main import D

DX_MM = 0.004
DT_S = 0.01

@njit
def dx_effective(x, y, tumor_pos, area, tk):
    """
    Calculate the effective dx value for points in the xy-plane.

    Parameters
    ----------
    x : foat
        Position in x direction
    y : float
        Position in y direction
    tumor_pos : np.ndarray
        ( m, (x,y) ) array with positions of centers of tumors.
    area : float
        Area of tumors. Same length unit as position.
    tk : np.ndarray
        (m,) array with tumor coefficients.

    Returns
    -------
    np.ndarray
        (x, y) array with effective dx value
    """
    m = tumor_pos.shape[0]
    tk_eff = 1.
    radius = np.sqrt(area / np.pi)

    for mi in range(m):
        tumor = tumor_pos[mi]
        x_dist = tumor[0] - x
        y_dist = tumor[1] - y
        pos = np.asarray((x_dist,y_dist))
        dist = np.linalg.norm(pos)
        if dist < radius:
            tk_eff *= tk[mi]

    return DX_MM * np.sqrt(tk_eff)

def brownian_N_2D_tumor(N, M, tumor_pos, tk, area, pr=0.5, pu=0.5):
    """
    Brownian motion for N particles with M steps in two dimensions.

    Parameters
    ----------
    N : int
        Number of particles
    M : int
        Number of moves.
    tumor_pos : np.ndarray
        ( m, (x,y) ) array with coordinates of tumor centers.
    tk : np.ndarray
        (m,) array with tumor coefficients.
    area : float
        Area of tumors (constant).
    pr : float
        Probability for taking a step to the right.
    py : float
        Probability for taking a step upwards.

    Returns
    -------
    np.array
        Time array, 1D.
    np.array
        Position array, 3D. Dimension ( M, N, (x,y) ).
    """
    assert 0. < pr < 1., 'Invalid probability pr'
    assert 0. < pu < 1., 'Invalid probability pu'

    positions = np.zeros((M, N, 2))
    random_values = np.random.uniform(0, 1, size=positions.shape)
    move_horizontal = np.random.uniform(0, 1, (M, N)) <= 0.5

    for t in range(1, M):
        for n in range(N):
            x, y = positions[t,n,0], positions[t,n,1]
            dx_eff: float = dx_effective(x, y, tumor_pos, area, tk)

            if move_horizontal[t,n]:
                if random_values[t,n,0] <= pr:
                    positions[t,n,0] = positions[t-1,n,0] + dx_eff
                else:
                    positions[t,n,0] = positions[t-1,n,0] - dx_eff
            else:
                if random_values[t,n,1] <= pu:
                    positions[t,n,1] = positions[t,n,1] + dx_eff
                else:
                    positions[t,n,1] = positions[t,n,1] - dx_eff

    return np.arange(M), np.cumsum(positions, axis=0)

def task_2a():
    print(f'D = {D(0.004,0.01):.1e}')

def task_2c():
    # Valus given in task
    N = 2
    M = 1000
    m = 15
    L = 0.02  # [mm]

    tumor_pos = np.random.uniform(0, L, size=(m,2))
    tk = np.full(shape=m, fill_value=0.1)
    area = np.pi * DX_MM ** 2
    t, pos = brownian_N_2D_tumor(N, M, tumor_pos, tk, area)

    # To plot tumors in colors / dx_eff
    xmin = np.min(pos[:,:,0])
    xmax = np.max(pos[:,:,0])
    ymin = np.min(pos[:,:,1])
    ymax = np.max(pos[:,:,1])
    nx, ny = 1000, 1000
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    dx = np.zeros((nx, ny))
    for xi in range(nx):
        for yi in range(ny):
            dx[xi,yi] = dx_effective(x[xi], y[yi], tumor_pos, area, tk)
    dx = np.transpose(dx)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    for n in range(N):
        plt.plot(pos[:,n,0], pos[:,n,1])

    c = plt.pcolormesh(x, y, dx, cmap='plasma_r')
    plt.colorbar(c)
    plt.show()

def main():
    # task_2a()
    task_2c()

if __name__ == '__main__':
    main()
