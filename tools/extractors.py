import cv2 as cv
import numpy as np
from skimage import feature
from sklearn.decomposition import PCA


def extract_hog(image):

    image: np.array = np.reshape(image, (35, 35))

    (H, hogImage) = feature.hog(image, orientations=16, pixels_per_cell=(4, 4),
	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
	visualize=True)

    return H, hogImage

def extract_hog_from_dataset(X):

    HOG = []
    for x in X:
        HOG.append(extract_hog(x)[0])
    return HOG

def contour(image):

    image: np.array = np.reshape(image, (35, 35)).astype('uint8')

    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    n = len(contours)

    if (len(contours) > 1):
        contours = sorted(contours, key = lambda x: cv.contourArea(x))
    
    return cv.contourArea(contours[-1]), n

def extract_contour_from_dataset(X):

    contours = []
    for x in X:
        contours.append(contour(x))
    return np.array(contours)

def extract_laplacian_from_dataset(X):

    laplacians = []
    for x in X:

        image: np.array = np.reshape(x, (35, 35)).astype('uint8')

        laplacians.append(cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5).flatten())

    return laplacians

def hu_moments(im):

    moments = cv.moments(im) 
    # Calculate Hu Moments 
    huMoments = cv.HuMoments(moments)

    return np.array(huMoments)

def extract_hu_moments_from_dataset(X):

    moments_hu = []
    for x in X:

        image: np.array = np.reshape(x, (35, 35)).astype('uint8')

        moments_hu.append(hu_moments(image).reshape(7))

    return np.array(moments_hu)

def extract_components_for_dataset(X):
    pca = PCA(n_components=50)
    X_ = pca.fit_transform(X)

    return X_
