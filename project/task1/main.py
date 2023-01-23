import numpy as np
import matplotlib.pyplot as plt

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
        Position array, 2D.
    """
    assert 0. < pr < 1., 'Invalid probability pr'

    random_values = np.random.uniform(0, 1, (M, N))
    steps = np.where(random_values <= pr, +1, -1)
    positions = np.cumsum(steps, axis=0)

    return np.arange(M), positions

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

    """
    We can assume the function is faster since it does not use for-loops. Using only numpy arrays, the machine can
    perform the calculations in C/C++ which makes them go faster than in native Python.
    """

def main():
    task_1c()
    task_1d()
    task_1e()

if __name__ == '__main__':
    main()
