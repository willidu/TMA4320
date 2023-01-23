import numpy as np
import matplotlib.pyplot as plt

def brownian_1d(M, pr=0.5):
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

def brownian_many_particles(N, M, pr=0.5):
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

    positions = np.zeros((N, M))
    random_values = np.random.uniform(0, 1, (N, M))
    for i in range(M):
        for j in range(N):
            if random_values[i,j] <= pr:
                positions[i,j] = positions[i-1,j] + 1.
            else:
                positions[i,j] = positions[i-1,j] - 1.

    return np.arange(M), positions

def task_1c():
    M = 10_000

    for pr in (0.45, 0.5, 0.55):
        plt.plot(*brownian_1d(M, pr), label=r'$p_r$ = '+f'{pr:.2f}')

    plt.legend()
    plt.xlabel('Step number')
    plt.ylabel('Position')
    plt.title(f'Brownian motion in 1D - {M = }')
    plt.show()

    # We observe that the movement favors the right direction if pr > 0.5, as expected.
    # M = 10 000 seems high enough to get "random" motion.

def task_1d():
    # Testing implementation in Task 1d.
    plt.plot(*brownian_many_particles(N=1000, M=1000, pr=0.5))
    plt.show()

def main():
    task_1c()
    task_1d()

if __name__ == '__main__':
    main()
