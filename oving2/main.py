import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.linalg import lu, solve


def lu_fac_native(A):
    """
    Calculate the LU factorization of a matrix A such that A = LU.

    Parameters
    ----------
    A : np.ndarray (M, N)
        Matrix with real elements to factorize.

    Returns
    -------
    L : np.ndarray (M, K)
        Lower triangular unit matrix.
    U : np.ndarray (K, N)
        Upper triangular matrix.
    """
    U = np.array(A, copy=True, dtype=float)

    assert U.shape[0] == U.shape[1], 'A is not quadratic'

    n, _ = U.shape
    L = np.eye(n, n)

    for k in range(0, n-1):
        for i in range(k+1, n):
            m_ik = U[i,k] / U[k,k]
            L[i,k] += m_ik

            for j in range(k, n):
                U[i,j] -= m_ik * U[k,j]

    return L, U


def lu_fac_pivot(A):
    """
    Calculate the LU factorization of a matrix A such that PA = LU.

    Parameters
    ----------
    A : np.ndarray (M, N)
        Matrix with real elements to factorize.

    Returns
    -------
    P : np.ndarray (M, M)
        Permutation matrix
    L : np.ndarray (M, K)
        Lower triangular unit matrix.
    U : np.ndarray (K, N)
        Upper triangular matrix.
    """
    U = np.array(A, copy=True, dtype=float)

    assert U.shape[0] == U.shape[1], 'A is not quadratic'

    n, _ = U.shape
    L = np.eye(n, n)
    s = np.max(np.abs(A), axis=1)  # Scaling vector
    p = np.arange(n)  # Permutation order vector

    for i in range(n):
        # Finding max element in row
        matr = np.abs(A)[p[i:],i] / s[p[i:]]
        max_index = np.argmax(matr) + i

        # Updating indicies
        temp = p[i]
        p[i] = p[max_index]
        p[max_index] = temp

    P = np.identity(n)[p]

    # Gauss elimination
    for k in range(0, n-1):
        for i in range(k+1, n):
            m_ik = U[p[i],k] / U[p[k],k]
            L[i,k] += m_ik

            for j in range(k, n):
                U[p[i],j] -= m_ik * U[p[k],j]

    return P, L, U[p]


def lu_fac_scipy(A):
    """
    Calculate the LU factorization of a matrix A such that A = LU.

    Parameters
    ----------
    A : np.ndarray (M, N)
        Matrix with real elements to factorize.

    Returns
    -------
    L : np.ndarray (M, K)
        Permuted lower triangular unit matrix.
    U : np.ndarray (K, N)
        Upper triangular matrix.
    P : (M, M)
        Permutation matrix.
    """
    P, L, U = lu(A, permute_l=False)  # Scipy returns matricies such that A = PLU
    assert_array_almost_equal(P@L@U, A)  # Checks A = PLU
    assert_array_almost_equal(P@A, L@U)  # Checks PA = LU
    return P @ L, U, P


def gauss_naive(A_matr, b_vec):
    """
    Solve the equation Ax = b using primitive Gauss Elimination.

    Parameters
    ----------
    A_matr : np.ndarray (N, N)
        Matrix with real elements.
    b_vec : np.ndarray (N,)
        Vector with real elements.

    Returns
    -------
    x : np.ndarray (N,)
        Solution to equation.
    """
    A = np.array(A_matr, copy=True, dtype=float)
    b = np.array(b_vec, copy=True, dtype=float)

    assert A.shape[0] == A.shape[1], 'A is not quadratic'
    assert A.shape[0] == b.shape[0], 'A and b do not have same first dimension'

    x = np.zeros_like(b)
    _, n = A.shape
    L = np.eye(n, n)

    for k in range(0, n-1):
        for i in range(k+1, n):
            m_ik = A[i,k] / A[k,k]
            L[i,k] += m_ik

            for j in range(k, n):
                A[i,j] -= m_ik * A[k,j]

            b[i] -= m_ik * b[k]

    x[-1] = b[-1] / A[-1,-1]
    for i in range(n-2, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += A[i,j] * x[j]
        x[i] = (b[i] - s) / A[i,i]

    return x


def gauss_pivot(A_matr, b_vec):
    A = np.array(A_matr, copy=True, dtype=float)
    b = np.array(b_vec, copy=True, dtype=float)

    assert A.shape[0] == A.shape[1], 'A is not quadratic'
    assert A.shape[0] == b.shape[0], 'A and b do not have same first dimension'

    x = np.zeros_like(b)
    s = np.max(np.abs(A), axis=1)  # Scaling vector
    n, _ = A.shape
    p = np.arange(n)  # Permutation order vector

    for i in range(n):
        # Finding max element in row
        matr = np.abs(A)[p[i:],i] / s[p[i:]]
        max_index = np.argmax(matr) + i

        # Updating indicies
        temp = p[i]
        p[i] = p[max_index]
        p[max_index] = temp

    for k in range(0, n-1):
        for i in range(k+1, n):
            m_ik = A[p[i],k] / A[p[k],k]

            for j in range(k, n):
                A[p[i],j] -= m_ik * A[p[k],j]

            b[p[i]] -= m_ik * b[p[k]]

    x[-1] = b[p[-1]] / A[p[-1],-1]
    for i in range(n-2, -1, -1):
        s = 0.
        for j in range(i+1, n):
            s += A[p[i],j] * x[j]
        x[i] = (b[p[i]] - s) / A[p[i],i]

    return x


def diagnoal_dominant(A):
    """ Checks if the matrix A is strictly diagnonally dominant. """
    return np.all(2. * np.diag(np.abs(A)) > np.sum(np.abs(A), axis=1))


def task1():
    A = np.asarray([
        [4, -8, 2],
        [5, 0, 10],
        [2, 2, 1]
    ], dtype=float)

    # Task 1a
    L, U = lu_fac_native(A)
    assert_array_almost_equal(L@U, A)
    print(f'Task 1a\nL = \n{L}\nU = \n{U}')

    # Task 1b
    P, L, U = lu_fac_pivot(A)
    print(f'Task 1b\nP = \n{P}\nPA = \n{P@A}\nLU = \n{L@U}')


def task2():
    # Testing 2a
    A = np.asarray([
        [10, 0, 20, 10],
        [4, -9, 2, 1],
        [8, 16, 6, 5],
        [2, 3, 2, 1]
    ], dtype=float)
    b = np.asarray([10, 2, 4, 3])

    x_anal = solve(A, b)
    x_naive = gauss_naive(A, b)
    x_pivot = gauss_pivot(A, b)

    assert_array_almost_equal(x_anal, x_naive)

    print(f'x (scipy.linalg.solve) {x_anal}')
    print(f'x (Gauss naive)        {x_naive}')
    print(f'x (Gauss pivot)        {x_pivot}')


def task3():
    A1 = np.asarray([
        [4, 2, -3],
        [2, 6, 1],
        [1, -2, 6]
    ], dtype=float)
    print(f'A1 is SDD: {diagnoal_dominant(A1)}')
    # Should be False

    A2 = np.asarray([
        [3, -1, 1],
        [2, -5, 1],
        [0, 1, 2]
    ], dtype=float)
    print(f'A2 is SDD: {diagnoal_dominant(A2)}')
    # Should be True

    A3 = np.asarray([
        [12, 1, -11, 0],
        [3, 7, -2, 1],
        [-2, 2, 6, -1],
        [1, -1, 2, 4]
    ], dtype=float)
    print(f'A3 is SDD: {diagnoal_dominant(A3)}')
    # Should be False


def main():
    task1()
    task2()
    task3()


if __name__ == '__main__':
    main()
