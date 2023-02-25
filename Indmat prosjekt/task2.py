import numpy as np
import matplotlib.pyplot as plt

from task1 import SVD_calculation, truncSVD, orthoproj

N_TRAIN = 1000
N_TEST = 200
ENMF_MAXITER = 50
PLOT_INT = 3  # Class. Change this to 4 if you would like to train on the number 4.

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
    d = np.arange(1, 784, 30) #Values for d
    image_index = 15  # Arbitrary image in A

    A = np.load('train.npy')[:,PLOT_INT,:N_TRAIN] / 255.0
    image = A[:,image_index]
    U, Z, Vt = SVD_calculation(A)
    normF = np.zeros(len(d))

    for i, d_value in enumerate(d):
        # For global digit
        W, H = truncSVD(U, Z, Vt, d=d_value)
        #Find the difference between a matrix A and its projections down on W (the truncated U-matrix from A's SVD)
        C = image - orthoproj(W, image)
        #Calculating the squared Frobenius norm of C
        normF[i] = np.sum(C**2)

    #Plotting results
    plt.semilogy(d, normF)
    plt.xlabel("d-values")
    plt.ylabel("Squared Frobenius norm of A - P_w(A)")
    plt.show()

if __name__ == '__main__':
    #task_2a()
    #task_2b()
    #task_2c()
    task_2d()
