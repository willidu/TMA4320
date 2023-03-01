import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from task1 import dist, orthoproj, nnproj, SVD_calculation, truncSVD
from task2 import plotimgs, ENMF_dict

N_TRAIN = 1024
N_TEST = 200
PLOT_DIGIT = 7

# Handed out code
def generate_test(test, digits = [0,1,2], N = 800):
    """
    Randomly generates test set.
    input:
        test: numpy array. Should be the test data loaded from file
        digits: python list. Contains desired integers
        N: int. Amount of test data for each class
    output:
        test_sub: (784,len(digits)*N) numpy array. Contains len(digits)*N images
        test_labels: (len(digits)*N) numpy array. Contains labels corresponding to the images of test_sub
    """
    assert N <= test.shape[2] , "N needs to be smaller than or equal to the total amount of available test data for each class"
    assert len(digits) <= 10, "List of digits can only contain up to 10 digits"

    # Arrays to store test set and labels
    test_sub = np.zeros((test.shape[0], len(digits)*N))
    test_labels = np.zeros(len(digits)*N)

    # Iterate over all digit classes and store test data and labels
    for i, digit in enumerate(digits):
        test_sub[:, i*N:(i+1)*N] = test[:,digit,:N]
        test_labels[i*N:(i+1)*N] = digit

    # Indexes to be shuffled 
    ids = np.arange(0,len(digits)*N)

    # Shuffle indexes
    np.random.shuffle(ids)

    # Return shuffled data 
    return test_sub[:,ids], test_labels[ids]

def get_distances(B, dict_list, SVD=True):
    """
    Calculate the distances of test data in matrix B by projecting onto
    different basises.

    Parameters
    ----------
    B : np.ndarray (784, n_train)
        Test dataset.
    dict_list : list[np.ndarray]
        List with dictionaries trained on different digits (and with different methods).
    SVD : bool
        Default False. Triggers either orthonormal (SVD) or non-negative (ENMF) projection
    
    Returns
    -------
    dist_list : list[np.ndarray]
        List with arrays containing projection distances for each basis.
    """
    if SVD:
        return np.asarray([dist(B, proj=orthoproj(W, B)) for W in dict_list])
    else:
        return np.asarray([dist(B, proj=nnproj(W, B)[1]) for W in dict_list])

def classification(B, dict_list, SVD=True):
    """
    Classifiy test data given different dictionaries based on projection distances.

    Parameters
    ----------
    B : np.ndarray (784, n)
        Test dataset.
    dict_list : list[np.ndarray]
        List with dictionaries trained on different digits (and with different methods).
    SVD : bool
        Default False. Triggers either orthonormal (SVD) or non-negative (ENMF) projection
    
    Returns
    -------
    np.ndarray (n,)
        Vector containing predicted class for all test data.
    """
    return np.argmin(get_distances(B, dict_list, SVD), axis=0)

def analyze_classification(test_labels, trained_labels, digits):
    """
    Perform classification on a test dataset.

    Parameters
    ----------
    test_labels : np.ndarray (n,)
        Class labels as integets 0-9.
    trained_labels : np.ndarray (n,)
        Known (correct) class labels as integers 0-9.
    digits : np.ndarray (num_digits,)
        Test digits.
    
    Returns
    -------
    accuracy : float
        Value between 0 and 1 as a total measure.
    recall : np.ndarray[float] (num_digits,)
        Accuracy measure for each class. Values > 0.
    """
    accuracy = np.count_nonzero(trained_labels == test_labels) / len(test_labels)  # Eq. (24)
    class_count = np.zeros_like(digits)     # Numerator eq. (25)
    class_possible = np.zeros_like(digits)  # Denomenator eq. (25)

    # Counting
    for a, b in zip(trained_labels, test_labels):
        if a == b:
            class_count[int(a)] += 1
        class_possible[int(b)] += 1
        
    recall = class_count / class_possible
    return accuracy, recall

def task_3b():
    # Loading training data
    train = np.load('train.npy')[:,:,:N_TRAIN] / 255.0
    train_digits = np.arange(5)

    # Training with SVD
    dict_list_SVD = [truncSVD(*SVD_calculation(matrix=train[:,i,:]), d=32)[0] for i in train_digits]

    # Training with ENMF
    dict_list_ENMF = [ENMF_dict(matrix=train[:,i,:], d=32) for i in train_digits]
    
    # Test setup
    # digits = [0,1,2]
    digits = train_digits.copy()
    test = np.load('test.npy') / 255.0

    # Handed out code
    A_test, A_labels = generate_test(test, digits=digits, N=N_TEST)
    print("Test data shape: ", A_test.shape)
    print("Test labels shape: ", A_labels.shape)
    print("First 16 labels: ", A_labels[:16])
    # plotimgs(A_test, nplot = 4)

    for dict_list, svd in zip((dict_list_SVD, dict_list_ENMF), (True, False)):
        print('\nSVD') if svd else print('\nENMF')

        # Classification
        classes = classification(A_test, dict_list, svd)

        # Analysis
        accuracy, recall = analyze_classification(A_labels, classes, digits)
        print(f'Accuracy : {accuracy:.3f}')
        print(f'Recall : {recall}')

def task_3f():
    """
    Change train_digits to fewer digits or change d values to lower runtime!
    """
    # Training
    train_digits = np.arange(5)  # Will train on 0 ... 9
    train = np.load('train.npy')[:,train_digits,:N_TRAIN] / 255.0

    # List with elements U1, Z1, Vt1, U2, Z2, ...
    SVD_dicts = [SVD_calculation(train[:,i,:]) for i in train_digits]

    # Testing
    test = np.load('test.npy') / 255.0
    test_digits = train_digits.copy()  # Will test all training digits
    A_test, A_labels = generate_test(test, digits=train_digits, N=N_TEST)

    d = np.logspace(1, 10, base=2, num=10, dtype=int)
    accuracies = np.zeros((10, 2))

    for i, d_value in enumerate(tqdm(d)):
        # Training
        truncSVD_dicts = [truncSVD(U, Z, Vt, d=d_value)[0] for U, Z, Vt in SVD_dicts]
        ENMF_dicts = [ENMF_dict(train[:,i,:], d=d_value) for i in train_digits]

        # Testing
        classes_SVD = classification(A_test, truncSVD_dicts, SVD=True)
        classes_ENMF = classification(A_test, ENMF_dicts, SVD=False)
        
        # Finding correct classifications
        accuracy_svd, _ = analyze_classification(classes_SVD, A_labels, digits=test_digits)
        accuracy_enmf, _ = analyze_classification(classes_ENMF, A_labels, digits=test_digits)

        accuracies[i] = accuracy_svd, accuracy_enmf
  
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.plot(d, accuracies[:,0], '-o', label='SVD')
    ax.plot(d, accuracies[:,1], '-o', label='ENMF')
    ax.legend(loc='lower right')
    ax.set(xlabel='$d$ value', ylabel='Accuracy [--]')
    plt.title('Accuracy for SVD and ENMF\n' + '$d = 2^i,\ i = 1,\dots,10$\n' + f'Digits: {train_digits}')
    plt.show()

if __name__ == '__main__':
    # task_3b()
    task_3f()
