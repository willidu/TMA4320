import numpy as np
import matplotlib.pyplot as plt

#Initialising testmatrices
A1 = np.array(((1000, 1), (0, 1), (0, 0)))
A2 = np.array(((1,0,0), (1,0,0), (0,0,1)))
B = np.array(((2,0,0), (1,0,1), (0,0,1)))

def SVD_calculation(matrix, printing=False, check=False):
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

U, Z, Vt = task_b(prints=True)
rank = np.linalg.matrix_rank(U)
print(rank)
    #task_b(A2, prints=True, checks=True)

