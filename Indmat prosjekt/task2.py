import numpy as np
import matplotlib.pyplot as plt

from task1 import SVD_calculation, truncSVD, orthoproj, nnproj

N_TRAIN = 1000
N_TEST = 200
PLOT_INT = 7  # Class. Change this to 4 if you would like to train on the number 4.

#Initialising testmatrix
B = np.asarray([
    [2, 0, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=float)

# Handed out function
def plotimgs(imgs, nplot = 4):
    """
    Plots the nplot*nplot first images in imgs on an nplot x nplot grid. 
    Assumes heigth = width, and that the images are stored columnwise
    
    Parameters
    ----------
    imgs : np.ndarray
        (height*width,N) array containing images, where N > nplot**2
    nplot : int
        nplot**2 images will be plotted
    """

    n = imgs.shape[1]
    m = int(np.sqrt(imgs.shape[0]))

    assert(n >= nplot**2), "Need amount of data in matrix N > nplot**2"

    # Initialize subplots
    fig, axes = plt.subplots(nplot,nplot)

    # Set background color
    plt.gcf().set_facecolor("lightgray")

    # Iterate over images
    for idx in range(nplot**2):

        # Break if we go out of bounds of the array
        if idx >= n:
            break

        # Indices
        i = idx//nplot
        j = idx%nplot

        axes[i,j].axis('off')
        axes[i,j].imshow(imgs[:,idx].reshape((m,m)), cmap = "gray")

    fig.tight_layout()
    plt.show()

def plot_projection(projections, d_values, image):
    """
    Plots the projection for different values of d.

    Parameters
    ----------
    projections : np.ndarray (len(d), 784)
        Array with different projections.
    d_values : np.ndarray (m,)
        Different d values used for projections.
    image : np.ndarray (784,)
        Original image used for projections.
    """
    fig, axes = plt.subplots(nrows=1, ncols=len(d_values)+1, figsize=(10,4))
    plt.gcf().set_facecolor("lightgray")

    axes = axes.flatten()  # flatten so they are easier to iterate over
    for i, ax in enumerate(axes):
        if i == 0:
            ax.imshow(image.reshape((28,28)), cmap='gray')
            ax.set(title='Original image')
        else:
            ax.imshow(projections[i-1].reshape((28,28)), cmap='gray')
            ax.set(title=f'd = {d_values[i-1]}')

        ax.axis('off')

    fig.tight_layout()
    plt.suptitle('Projection on dictionary ' + r'$W=U_d$', fontsize=20)
    plt.show()

def ENMF_dict(matrix, d):
    """
    Calculate an exemplar-based non-negative dictionary.

    Parameters
    ----------
    matrix : np.ndarray (m, n)
    d : int
        Number of columns to be selected. Needs to be <= n.

    Returns
    -------
    W : np.ndarray (m, d)
        Non-negative trained dictionary.

    Raises
    ------
    ValueError. If d > n. 
    """
    W = np.array(matrix, copy=True, dtype=float)
    n = W.shape[1]

    if n == d:
        return W

    elif d < n:
        rng = np.random.default_rng()
        return rng.choice(W, size = d, axis = 1, replace = False)

    else:
        raise ValueError('Invalid d value, must be less than number of columns in W')

def task_2a():
    train = np.load('train.npy')[:,:,:N_TRAIN] / 255.0
    plotimgs(train[:,PLOT_INT,100:])

def task_2b():
    # Loading the first N pictures of digit PLOT_INT
    A = np.load('train.npy')[:,PLOT_INT,:N_TRAIN] / 255.0
    U, Z, Vt = SVD_calculation(A)
    W, H = truncSVD(U, Z, Vt, d=16)
    plotimgs(W)

    singular_values = np.diag(Z)
    rank = np.linalg.matrix_rank(Z)

    plt.semilogy(np.arange(rank), singular_values[:rank])
    plt.title(f'Singular values 0 - {rank}')
    plt.xlabel('Value nr.')
    plt.ylabel('Value')
    plt.xlim(0, rank)
    plt.show()

    # Observing that the first numbers have the highest value, which corresponds to more important
    # to reconstruct A

def task_2c():
    """
    Here we do a lot of stuff for two different digits. TODO
    One is set at the top of the file (named PLOT_INT)
    The other is chosen to be either 0 or 1.
    """
    d = (16, 32, 64, 128)
    image_index = 15  # Arbitrary image in A and B

    if PLOT_INT == 0:
        other_digit = 1  # If global plot digit is zero, we chose another digit
    else:
        other_digit = 0

    # Remember to only load and calculate SVD once in the notebook
    A = np.load('train.npy')[:,PLOT_INT,:N_TRAIN] / 255.0

    U, Z, Vt = SVD_calculation(A)

    image = A[:,image_index]
    image_other = np.load('train.npy')[:,other_digit,image_index] / 255.0

    projections = np.zeros((len(d), A.shape[0]))
    projections_other = np.zeros_like(projections)

    for i, d_value in enumerate(d):
        # For global digit
        W, H = truncSVD(U, Z, Vt, d=d_value)
        projections[i] = orthoproj(W, image)

        # For other digit
        projections_other[i] = orthoproj(W, image_other)

    # Plotting projections and original image
    plot_projection(projections, d, image)
    plot_projection(projections_other, d, image_other)

def task_2d():
    N = 10
    d = np.arange(1, 784, N) #Values for d

    if PLOT_INT == 0:
        other_digit = 1  # If global plot digit is zero, we chose another digit
    else:
        other_digit = 0

    A = np.load('train.npy')[:,PLOT_INT,:N_TRAIN] / 255.0
    B = np.load('train.npy')[:,other_digit,:N_TRAIN] / 255.0
    
    U, Z, Vt = SVD_calculation(A)

    #Creating y-vectors
    normF_1 = np.zeros_like(d)
    normF_2 = np.zeros_like(d)
    
    for i, d_value in enumerate(d):
        W, _ = truncSVD(U, Z, Vt, d=d_value)

        # Projecting and difference
        C1 = A - orthoproj(W, A)
        C2 = B - orthoproj(W, B)

        #Calculating the squared Frobenius norm of C
        normF_1[i] = np.sum(C1**2)
        normF_2[i] = np.sum(C2**2)

    #Plotting results
    plt.semilogy(d, normF_2, label="Non-trained digit")
    plt.semilogy(d, normF_1, label="Trained digit")
    plt.title("Frobenius norm: SVD approach")
    plt.xlabel(f"d-values, steplength={N}")
    plt.ylabel("Squared Frobenius norm of A - P_w(A)")
    plt.legend()
    plt.show()

def task_2e():
    A = np.load('train.npy')[:,PLOT_INT,:N_TRAIN] / 255.0

    # ENMF approach
    W = ENMF_dict(A, d=32)
    _, P = nnproj(W, A)
    plotimgs(P)

    # SVD for comparison
    W, _ = truncSVD(*SVD_calculation(A), d=32)
    P = orthoproj(W, A)
    plotimgs(P)

def task_2f():
    d = np.logspace(1, 3, num=10, dtype=int)

    if PLOT_INT == 0:
        other_digit = 1  # If global plot digit is zero, we chose another digit
    else:
        other_digit = 0

    A = np.load('train.npy')[:,PLOT_INT,:N_TRAIN] / 255.0
    B = np.load('train.npy')[:,other_digit,:N_TRAIN] / 255.0

    #Creating y-vectors
    normF_1 = np.zeros_like(d)
    normF_2 = np.zeros_like(d)
    
    for i, d_value in enumerate(d):
        W = ENMF_dict(A, d_value)

        # Projection
        _, proj1 = nnproj(W, A)
        _, proj2 = nnproj(W, B)

        C1 = A - proj1
        C2 = B - proj2

        #Calculating the squared Frobenius norm of C
        normF_1[i] = np.sum(C1**2)
        normF_2[i] = np.sum(C2**2)

    #Plotting results
    plt.semilogy(d, normF_2, label="Non-trained digit")
    plt.semilogy(d, normF_1, label="Trained digit")
    plt.title("Frobenius norm: EMNF approach")
    plt.xlabel(f"d-values")
    plt.ylabel("Squared Frobenius norm of A - P_w(A)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    task_2a()
    task_2b()
    task_2c()
    task_2d()
    task_2e()
    task_2f()
