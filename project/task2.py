"""
All lengths are in millimeters!
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy import ndimage

from task1 import D

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

@njit
def brownian_N_2D_tumor(N, M, tumor_pos, tk, area, L=None, pr=0.5, pu=0.5):
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
    L : float
        Length of bounding box (0, L) x (0, L).
    pr : float
        Probability for taking a step to the right.
    pu : float
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
    if L is not None:
        assert L > 0., 'Invalid value L'

    positions = np.zeros((M, N, 2))
    random_values = np.random.uniform(0, 1, size=(M,N))
    move_horizontal = np.random.uniform(0, 1, M) <= 0.5

    for t in range(0, M-1):
        for n in range(N):
            x, y = positions[t,n,0], positions[t,n,1]
            dx_eff: float = dx_effective(x, y, tumor_pos, area, tk)

            if move_horizontal[t]:
                # Keep y coordinate
                positions[t+1,n,1] = positions[t,n,1]

                # Update x coordinate
                if random_values[t,n] <= pr:
                    positions[t+1,n,0] = positions[t,n,0] + dx_eff
                else:
                    positions[t+1,n,0] = positions[t,n,0] - dx_eff
            else:
                # Keep x coordinate
                positions[t+1,n,0] = positions[t,n,0]

                #Update y coordinate
                if random_values[t,n] <= pu:
                    positions[t+1,n,1] = positions[t,n,1] + dx_eff
                else:
                    positions[t+1,n,1] = positions[t,n,1] - dx_eff

            # Periodic boundary conditions
            if L is not None:
                positions[t+1,n] -= np.floor(positions[t+1,n] / L) * L

    return np.arange(M), positions

def sobel(im):
    """
    Applies Sobel filter on greyscale image.

    Parameters
    ----------
    im : np.ndarray[int]
        (nx, ny) Array containing values for greyscale image.

    Returns
    -------
    Sx : np.ndarray[int]
        (nx, ny) Sobel filter result in X direction.
    Sy : np.ndarray[int]
        (nx, ny) Sobel filter result in Y direction.
    mag : np.ndarray[int]
        (nx, ny) Sobel filter normalized magnitude.
    """
    Sx = ndimage.sobel(im, axis=0)
    Sy = ndimage.sobel(im, axis=1)
    magnitude = np.sqrt(Sx ** 2 + Sy ** 2)
    magnitude *= 255. / np.max(magnitude)  # Normalizing
    return Sx, Sy, magnitude

def plot_sobel(im, show=False):
    """
    Apply Sobel filter on a greyscale image and plot the result.

    Parameters
    ----------
    im : np.ndarray[int]
        (nx, ny) Array containing values for greysacle image.
    show : bool
        Default False. Triggers plt.show().

    Returns
    -------
    plt.figure()
        Subplot figure.
    """
    fig, ax = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=(9, 6),
        subplot_kw={'xticks': [], 'yticks': []}  # Removes axes ticks and numbers
    )
    cmap = 'Greys_r'  # Greysacle colormap
    Sx, Sy, mag = sobel(im)

    # Plotting original figure
    ax[0,0].imshow(im, cmap)
    ax[0,0].set(title='Original')

    # Plotting Sobel x direction
    ax[0,1].imshow(Sx, cmap)
    ax[0,1].set(title='X')

    # Plotting Sobel y direction
    ax[1,0].imshow(Sy, cmap)
    ax[1,0].set(title='Y')

    # Plotting Sobel magnitude
    ax[1,1].imshow(mag, cmap)
    ax[1,1].set(title='S')

    plt.suptitle('Sobel filter', size='x-large', weight='bold')
    plt.tight_layout()

    if show:
        plt.show()

    return fig

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
    xmin, xmax = -DX_MM, L + DX_MM
    ymin, ymax = -DX_MM, L + DX_MM

    # Calculating DX for color plot
    n = 200
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    dx = np.zeros((n,n))
    for xi in range(n):
        for yi in range(n):
            dx[yi,xi] = dx_effective(x[xi], y[yi], tumor_pos, area, tk)

    c = plt.pcolormesh(x, y, dx, cmap='plasma_r')
    plt.colorbar(c)

    # Plotting random walk
    for n in range(N):
        plt.plot(pos[:,n,0], pos[:,n,1])

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title('Task 2c')
    plt.show()

def test_periodic():
    N = 2
    M = 1000
    m = 15
    L = 0.02  # [mm]

    xmin, xmax = 0, L
    ymin, ymax = 0, L

    tumor_pos = np.random.uniform(xmin, xmax, size=(m,2))
    tk = np.full(shape=m, fill_value=0.1)
    area = np.pi * DX_MM ** 2
    t, pos = brownian_N_2D_tumor(N, M, tumor_pos, tk, area, L)

    # Calculating DX for color plot
    n = 300
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    dx = np.zeros((n,n))
    for xi in range(n):
        for yi in range(n):
            dx[yi,xi] = dx_effective(x[xi], y[yi], tumor_pos, area, tk)

    c = plt.pcolormesh(x, y, dx, cmap='plasma_r')
    plt.colorbar(c)

    # Plotting random walk
    for n in range(N):
        plt.scatter(pos[:,n,0], pos[:,n,1])

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title('Periodic boundary conditions')
    plt.show()

def test_plot_sobel():
    im_arr = plt.imread('sobel_test_fig.jpg').astype('int32')
    fig = plot_sobel(im_arr)
    plt.show()

def main():
    task_2a()
    task_2c()
    test_periodic()
    test_plot_sobel()

if __name__ == '__main__':
    main()
