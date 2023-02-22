import numpy as np
import matplotlib.pyplot as plt

#Initialising testmatrices
A1 = np.array(((1000, 1), (0, 1), (0, 0)))
A2 = np.array(((1,0,0), (1,0,0), (0,0,1)))
B = np.array(((2,0,0), (1,0,1), (0,0,1)))

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
    U : np.ndarray
        Orthogonal matrix with singular vectors of A.
    Z : np.ndarray
        shape=(n,n). Diagonal matrix containing singular values of A.
    Vt : np.ndarray
        Orthogonal matrix
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    Z = np.diag(S)

    if printing:
        print(f"A = \n {matrix}" + "\n")
        print(f"U = \n {U}" + "\n")
        print(f"Zigma =\n {Z}" + "\n")
        print(f"V_transponert =\n {Vt}" + "\n")
    if check:
        A_check = U@Z@Vt
        print(f"U*Z*V_transponert = \n {A_check}")

    return U, Z, Vt

def task_a(prints=False, checks=False):
    U, Z, Vt = SVD_calculation(A1, printing=prints, check=checks)
    #Discussion: Which of the basis vectors in U (=W1) is the most important one for reconstructing A1?
    return U, Z, Vt

def task_b(prints=False, checks=False):
    U, Z, Vt = SVD_calculation(A2, printing=prints, check=checks)
    return U, Z, Vt

def truncSVD(U, Z, Vt, d):

    return

