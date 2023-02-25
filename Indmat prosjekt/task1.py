import numpy as np
import matplotlib.pyplot as plt

#Initialising testmatrices
A1 = np.array(((1000, 1), (0, 1), (0, 0)))
A2 = np.array(((1,0,0), (1,0,0), (0,0,1)))
B = np.array(((2,0,0), (1,0,1), (0,1,0)))

def SVD_calculation(matrix, printing=False, check=False):
    """
    This function calculates the matrices U, Z (Zigma), and V (transposed) by performing SVD on input matrix.
    It can also print the matrices and calculate the dot-product of them.

    Parameters
    ----------
    matrix : np.ndarray
        Input matrix that the function will calculate the SVD of.
    printing : bool, defualt=False. 
        If set to True, the function prints out all matrices (U, Z, V) from the SVD calculation.
    check : bool
        If set to True, the function calculates the dot-product of U, Z, and Vt which should give us our input matrix
        we started with.

    Returns
    -------
    U : np.ndarray (m, d)
        Left singular matrix where columns contain the eigenvectors of A*A.T (testmatrix * testmatrix.transposed).
    Z : np.ndarray (d, d)
        Diagonal matrix containing singular values.
    Vt : np.ndarray (d, d)
        Right singular matrix where columns contain the eigenvectors of A.T*A (testmatrix.transposed * testmatrix).
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    Z = np.diag(S)

    return U, Z, Vt

def printAndCheck(A, U, Z, Vt):
    """
    Prints matrices. Checks if their product is equal to desired matrix.

    Parameters
    ----------
    A : np.ndarray (m, d)
        Desired matrix. 
    U : np.ndarray (m, d)
        Left singular matrix where columns contain the eigenvectors of A*A.T (testmatrix * testmatrix.transposed).
    Z : np.ndarray (d, d)
        Diagonal matrix containing singular values.
    Vt : np.ndarray (d, d)
        Right singular matrix where columns contain the eigenvectors of A.T*A (testmatrix.transposed * testmatrix).

    """
    print(f"Original matrix = \n {A}" + "\n")
    print(f"U = \n {U}" + "\n")
    print(f"Zigma =\n {Z}" + "\n")
    print(f"V_transponert =\n {Vt}" + "\n")
    
    A_check = U@Z@Vt
    print(f"U*Z*V_transponert = \n {A_check}")
    np.testing.assert_almost_equal(A_check, A, decimal=5, err_msg="They are not equal to five decimalpoints.", verbose=True)


def orthoproj(W, B):
    """
    Calculate the orthogonal prjection of colums in matrix B onto the basis in W.
    Eq. (14).

    Parametes
    ---------
    W : np.ndarray (m, d)
        Basis dictionary with orthonormal colum vectors to be projected onto.
    B : np.ndarray, (d, n)
        Idk a matrix
    """
    return W @ (W.T @ B)

def dist(B, proj):
    """
    Columnwise distance between projected vectors and the vector space. (?)
    Eq. (15), (16)

    Parametes
    ---------
    TODO
    """
    return np.linalg.norm(B - proj, ord=2, axis=0)

def truncSVD(U, Z, Vt, d, verbose=False, test=False):
    """
    Calculate the truncated SVD given the SVD factorization of a matrix.

    Parameters
    ----------
    U : np.ndarray (m, d)
        Left singular matrix where columns contain the eigenvectors of A*A.T (testmatrix * testmatrix.transposed).
    Z : np.ndarray (d, d)
        Diagonal matrix containing singular values.
    Vt : np.ndarray (d, d)
        Right singular matrix where columns contain the eigenvectors of A.T*A (testmatrix.transposed * testmatrix).
    d : int
        Number of singular values (<= n).
    verbose : bool
        Default=False. If True, print SVD matrices.
    test : bool
        Default=False. If True, print the matrix product of SVD matrices.

    Returns
    -------
    W : np.ndarray (m, d)
        Truncated dictionary.
    H : np.ndarray (d, n)
        Truncated weights
    """
    # Truncating
    U = U[:,:d]
    Z = Z[:d,:d]
    Vt = Vt[:d,:]

    if verbose:
        print(f"U = \n {U}" + "\n")
        print(f"Sigma =\n {Z}" + "\n")
        print(f"V^T =\n {Vt}" + "\n")
    if test:
        print(f"U*Z*V_transponert = \n {U @ Z @ Vt}")

    # W = U, H = Sigma V^T
    return U, Z @ Vt

def task_a(prints=False, checks=False):
    U, Z, Vt = SVD_calculation(A1)
    printAndCheck(A1, U, Z, Vt)
    #Discussion: Which of the basis vectors in U (=W1) is the most important one for reconstructing A1?
    #Answer: The first(s) columns contain the most information about A as they will be multiplied with the largest
    #singular values

def task_b(prints=False, checks=False):
    U, Z, Vt = SVD_calculation(A2)
    printAndCheck(A2, U, Z, Vt)
    # Observing that the last singular value is 0, so it can be removed
    W, H = truncSVD(U, Z, Vt, d=2, verbose=True, test=True)

def task_c():
    # Test matrix A1
    U, Z, Vt = SVD_calculation(A1)
    Pw = orthoproj(W=U, B=B)
    print(dist(B, Pw))
    # -> [0, 1, 0]. Ok.

    # Test matrix A2
    U, Z, Vt = SVD_calculation(A2)
    Pw = orthoproj(W=U, B=B)
    print(dist(B, Pw))
    # -> [0, 0, 0]. Feil?

def ENMF_calculation(matrix, d, maxiter=50, delta=1e-10):
    A  = matrix.copy()
    n = A.shape[1]
    
    if n == d:
        W = A
    else:
        rng = np.random.default_rng()
        W = rng.choice(A, size = d, axis = 1, replace = False)

    H = np.random.uniform(0, 1, (d, n))
    matrix_1 = W.T @ A
    matrix_2 = W.T @ W

    for _ in range(maxiter):
        H = H * matrix_1 / (matrix_2 @ H + delta)

    return W, H

def nnproj(W, H):
    return W @ H

def task_d():
    print(A1)
    W, H = ENMF_calculation(A1, d=2)
    P = nnproj(W, H)
    print(P)
    dists = dist(B, P)
    print(dists)

if __name__ == '__main__':
    # task_b(True, True)
    # task_c()
    task_d()
