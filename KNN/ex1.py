# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from copy import deepcopy

# initialze the coentroids
def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0.        , 0.        , 0.        ],
                            [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                            [0.49019608, 0.41960784, 0.33333333],
                            [0.02745098, 0.        , 0.        ],
                            [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                            [0.14509804, 0.12156863, 0.12941176],
                            [0.4745098 , 0.40784314, 0.32941176],
                            [0.00784314, 0.00392157, 0.02745098],
                            [0.50588235, 0.43529412, 0.34117647],
                            [0.09411765, 0.09019608, 0.11372549],
                            [0.54509804, 0.45882353, 0.36470588],
                            [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                            [0.4745098 , 0.38039216, 0.33333333],
                            [0.65882353, 0.57647059, 0.49411765],
                            [0.08235294, 0.07843137, 0.10196078],
                            [0.06666667, 0.03529412, 0.02352941],
                            [0.08235294, 0.07843137, 0.09803922],
                            [0.0745098 , 0.07058824, 0.09411765],
                            [0.01960784, 0.01960784, 0.02745098],
                            [0.00784314, 0.00784314, 0.01568627],
                            [0.8627451 , 0.78039216, 0.69803922],
                            [0.60784314, 0.52156863, 0.42745098],
                            [0.01960784, 0.01176471, 0.02352941],
                            [0.78431373, 0.69803922, 0.60392157],
                            [0.30196078, 0.21568627, 0.1254902 ],
                            [0.30588235, 0.2627451 , 0.24705882],
                            [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None

def assignment(X, centroids, clusters):
    for i in range(len(X)):
            # calculate the distances from the point to all of the centroids
            distances = np.linalg.norm(X[i] - centroids, axis = 1) ** 2
            # find the closest centroid
            cluster = np.argmin(distances)
            # assign the point to the closest centroid
            clusters[i] = cluster
        

def update_cent(X, centroids, k, clusters):
    for i in range(k):
            # get all the assigned points to the specific centroid
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            # calculate the mean of the points and assign it as the new centroid
            centroids[i] = np.mean(points, axis = 0)


def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]

def kmeans(X, k):
    print ('k={}:'.format(k))
    
    # initial the centroids
    centroids = init_centroids(X, k)
    
    # store the value of the centroids before update to check if a change occured
    cent_old = np.zeros(centroids.shape)
    
    # cluster labels
    clusters = np.zeros(len(X), dtype = int)
    
    # distance between new and old centroids
    error = np.linalg.norm(centroids - cent_old, axis = 1)
    
    stop = 0
    
    # stops after 10 iterations
    while (error != 0).any() or stop < 11:
        print ('iter {}: '.format(stop) + print_cent(centroids))
        # assigning the points to the closest centroid
        assignment(X, centroids, clusters)
        
        # store the last centroid positions    
        cent_old = deepcopy(centroids)
        
        # finding the new centroids
        update_cent(X, centroids, k, clusters)
        
        # re-evaluate the distance between the old and new centroids
        error = np.linalg.norm(centroids - cent_old, axis = 1) ** 2
        stop = stop + 1
        if stop == 11:
            break
    
def main():
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])    
    
    # executing k-means for 2,4,8,16 clusters
    k = 2
    for i in range(4):
        kmeans(X, k)
        k = k * 2

main()
