import numpy as np

from task1 import dist, orthoproj, nnproj, SVD_calculation, truncSVD
from task2 import plotimgs, ENMF_dict

N_TRAIN = 1000
N_TEST = 200

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
            class_count[a] += 1
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

if __name__ == '__main__':
    task_3b()
