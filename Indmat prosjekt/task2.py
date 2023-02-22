import numpy as np
import matplotlib.pyplot as plt

from task1 import SVD_calculation

N_TRAIN = 1000
N_TEST = 200
ENMF_MAXITER = 50
PLOT_INT = 7  # Class. Change this to 4 if you would like to train on the number 4.

# Handed out function
def plotimgs(imgs, nplot = 4):
    """
    Plots the nplot*nplot first images in imgs on an nplot x nplot grid. 
    Assumes heigth = width, and that the images are stored columnwise
    input:
        imgs: (height*width,N) array containing images, where N > nplot**2
        nplot: integer, nplot**2 images will be plotted
    """

    n = imgs.shape[1]
    m = int(np.sqrt(imgs.shape[0]))

    assert(n > nplot**2), "Need amount of data in matrix N > nplot**2"

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

def task_2a():
    train = np.load('train.npy')[:,:,:N_TRAIN] / 255.0
    plotimgs(train[:,PLOT_INT,100:])

def task_2b():
    # Loading the first N pictures of digit PLOT_INT
    A = np.load('train.npy')[:,PLOT_INT,:N_TRAIN] / 255.0
    U, Z, Vt = SVD_calculation(A)
    plotimgs(U)

    singular_values = np.diag(Z)
    rank = np.linalg.matrix_rank(Z)
    print(rank)

    plt.semilogy(np.arange(rank), singular_values[:rank])
    plt.title(f'Singular values 0 - {rank}')
    plt.xlabel('Value nr.')
    plt.ylabel('Value')
    plt.xlim(0, rank)
    plt.show()

    # Observing that the first numbers have the highest value, which corresponds to more important
    # to reconstruct A

if __name__ == '__main__':
    # task_2a()
    task_2b()
