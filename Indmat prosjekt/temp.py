def printAndCheck(A, U, Z, Vt, verbose=False, check=False):
    if verbose:
        print(f"A = \n {A}" + "\n")
        print(f"U = \n {U}" + "\n")
        print(f"Zigma =\n {Z}" + "\n")
        print(f"V_transponert =\n {Vt}" + "\n")
    if check:
        A_check = U@Z@Vt
        print(f"U*Z*V_transponert = \n {A_check}")

def task_a(prints=False, checks=False):
    U, Z, Vt = SVD_calculation(A1)
    printAndCheck(A1, U, Z, Vt, verbose=prints, check=checks)
    #Discussion: Which of the basis vectors in U (=W1) is the most important one for reconstructing A1?
    #Answer: The first(s) columns contain the most information about A as they will be multiplied with the largest
    #singular values
    return U, Z, Vt

def task_b(prints=False, checks=False):
    U, Z, Vt = SVD_calculation(A2)
    printAndCheck(A2, U, Z, Vt, verbose=prints, check=checks)
    UTrunc, ZTrunc, VtTrunc = truncSVD(U, Z, Vt, 2)
    printAndCheck(A2, UTrunc, ZTrunc, VtTrunc, verbose=True)