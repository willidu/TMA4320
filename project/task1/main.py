import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit

def D(dx=1., dt=1.):
    """
    Gives the constant in Eq. 5.
    """
    return dx ** 2 / (2 * dt)

def brownian_single_1D(M, pr=0.5):
    """
    Brownian motion for a single particle with M steps.

    Parameters
    ----------
    M : int
        Number of moves.
    pr : float
        Probability for taking a step to the right.

    Returns
    -------
    np.array
        Time array, 1D.
    np.array
        Position array, 1D.
    """
    assert 0. < pr < 1., 'Invalid probability pr'

    positions = np.zeros(M+1)
    random_values = np.random.uniform(0, 1, M)

    for i, value in enumerate(random_values):
        if value <= pr:
            positions[i+1] = positions[i] + 1.
        else:
            positions[i+1] = positions[i] - 1.

    return np.arange(M+1), positions

def brownian_N_1D(N, M, pr=0.5):
    """
    Brownian motion for N particles with M steps.

    Parameters
    ----------
    N : int
        Number of particles
    M : int
        Number of moves.
    pr : float
        Probability for taking a step to the right.

    Returns
    -------
    np.array
        Time array, 1D.
    np.array
        Position array, 2D.
    """
    assert 0. < pr < 1., 'Invalid probability pr'

    positions = np.zeros((M, N))
    random_values = np.random.uniform(0, 1, (M, N))
    for i in range(M):
        for j in range(N):
            if random_values[i,j] <= pr:
                positions[i,j] = positions[i-1,j] + 1.
            else:
                positions[i,j] = positions[i-1,j] - 1.

    return np.arange(M), positions

def brownian_N_1D_vectorized(N, M, pr=0.5):
    """
    Brownian motion for N particles with M steps.

    Parameters
    ----------
    N : int
        Number of particles
    M : int
        Number of moves.
    pr : float
        Probability for taking a step to the right.

    Returns
    -------
    np.array
        Time array, 1D.
    np.array
        Position array, 2D. (M, N)
    """
    assert 0. < pr < 1., 'Invalid probability pr'

    random_values = np.random.uniform(0, 1, (M, N))
    steps = np.where(random_values <= pr, +1, -1)
    positions = np.cumsum(steps, axis=0)

    return np.arange(M), positions

def brownian_N_2D(N, M, pr=0.5, py=0.5):
    """
    Brownian motion for N particles with M steps in two dimensions.

    Parameters
    ----------
    N : int
        Number of particles
    M : int
        Number of moves.
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
    assert 0. < py < 1., 'Invalid probability py'

    positions = np.zeros((M, N, 2))
    random_values = np.random.uniform(0, 1, size=positions.shape)
    steps_x = np.where(random_values[:,:,0] < pr, +1, -1)
    steps_y = np.where(random_values[:,:,1] < py, +1, -1)
    positions[:,:,0] += steps_x
    positions[:,:,1] += steps_y

    return np.arange(M), np.cumsum(positions, axis=0)

@njit
def count_zeros_N_1D(positions: np.ndarray):
    """
    Calculates n(t) for N particles in 1D.

    Parameters
    ----------
    positions : np.ndarray, (M, N)
        Positional array with time as first axis.

    Returns
    -------
    np.ndarray, (M,)
        Time array.
    np.ndarray, (M,)
        Array with n(t) values.
    """
    M, N = positions.shape
    n = np.zeros(M)
    been_to_origo_indicies = []

    for t in range(1, M):
        count = 0
        for i in range(N):
            if positions[t,i] == 0:
                if not (i in been_to_origo_indicies):
                    been_to_origo_indicies.append(i)
                    count += 1

        n[t] = n[t-1] + count / N

    return np.arange(M), n

@njit
def count_zeros_N_2D(positions: np.ndarray):
    """
    Calculates n(t) for N particles in 2D.

    Parameters
    ----------
    positions : np.ndarray, (M, N, (x,y))
        Positional array with time as first axis.

    Returns
    -------
    np.ndarray, (M,)
        Time array.
    np.ndarray, (M,)
        Array with n(t) values.
    """
    M, N, _ = positions.shape
    n = np.zeros(M)
    been_to_origo_indicies = []

    for t in range(1, M):
        count = 0
        for i in range(N):
            in_origo = positions[t,i,0] == 0 and positions[t,i,1] == 0
            if in_origo:
                if not (i in been_to_origo_indicies):
                    been_to_origo_indicies.append(i)
                    count += 1

        n[t] = n[t-1] + count / N

    return np.arange(M), n

def task_1c():
    M = 10_000

    for pr in (0.45, 0.5, 0.55):
        plt.plot(*brownian_single_1D(M, pr), label=r'$p_r$ = '+f'{pr:.2f}')

    plt.legend()
    plt.xlabel('Step number')
    plt.ylabel('Position')
    plt.title(f'Brownian motion in 1D - {M = }')
    plt.show()

    # We observe that the movement favors the right direction if pr > 0.5, as expected.
    # M = 10 000 seems high enough to get "random" motion.

def task_1d():
    # Testing implementation in Task 1d.
    plt.plot(*brownian_N_1D(N=100, M=1000, pr=0.5))
    plt.title('Task 1d')
    plt.show()

def task_1e():
    # Testing implementation in Task 1e.
    # TODO ad timing
    plt.plot(*brownian_N_1D_vectorized(N=100, M=1000, pr=0.5))
    plt.title('Task 1e')
    plt.show()

    # We can assume the function is faster since it does not use for-loops. Using only numpy arrays, the machine can
    # perform the calculations in C/C++ which makes them go faster than in native Python.

def task_1f():
    t, _ = brownian_N_1D_vectorized(N=1000, M=1000, pr=0.5)
    sigma_sq = 2 * D() * t

    plt.plot(t, sigma_sq)
    plt.title(r'Empirical variance, $\Delta x = 1 = \Delta t$')
    plt.xlabel('Time steps')
    plt.ylabel(r'Variance $\sigma^2$')

    # Gives a straight line, as one would excpect from 1a.

    a, b = curve_fit(lambda x, a, b: a*x + b, xdata=t, ydata=sigma_sq)[0]
    print(f'Linear fit: a = {a:.3e}, b = {b:.3e}')

    # We observe that the slope is equal to 1 and the y intercept is approximately 10^-12.
    # Dont know if higher N or M will improve stuff.

    plt.show()

def task_1g():
    N = 4
    M = 1000
    # Isotrop system
    t, pos = brownian_N_2D(N, M)

    # For every atom, plot the scatter plot of position in x and y for all time points
    for n in range(N):
        plt.scatter(pos[:,n,0], pos[:,n,1])

    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title(f'Brownian motion in 2D \nN = 4, M = 1000, ' + r'$p_r = 0.5 = p_u$')
    plt.show()

    # Non-isotrop system
    t, pos = brownian_N_2D(N, M, pr=0.65, py=0.35)

    # For every atom, plot the scatter plot of position in x and y for all time points
    for n in range(N):
        plt.scatter(pos[:,n,0], pos[:,n,1])

    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title(f'Brownian motion in 2D \nN = 4, M = 1000, ' + r'$p_r = 0.65, p_u = 0.35$')
    plt.show()

def task_1i():
    N = 100
    M = 100_000

    # Isotrop system
    _, pos = brownian_N_1D_vectorized(N, M)
    t, n = count_zeros_N_1D(pos)
    plt.plot(t, n, label='1D')

    # Non-isotrop system
    _, pos = brownian_N_2D(N, M)
    t, n = count_zeros_N_2D(pos)
    plt.plot(t, n, label='2D')

    plt.xlabel('Time steps')
    plt.ylabel('n(t)')
    plt.title(f'{N = }, {M = }')
    plt.axhline(y=1, ls='--', c='k', alpha=0.5)
    plt.show()

    # 1D Looks correct, converges quite fast
    # 2D also looks correct, but converges a lot slower
    # Ran simulations with higher N and M for 2D but run time was waaaay too long
    # Higher N does not affect convergence, only M does.
    # Therefore n(t, M) is a good approximation of P(x=0, t->inf) for high M.
    # Convergence also changes quite a lot from run to run, suggesting that M=100_000 is too low for 2D.

def main():
    task_1c()
    task_1d()
    task_1e()
    task_1f()
    task_1g()
    task_1i()

if __name__ == '__main__':
    main()
