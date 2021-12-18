import cv2
from numpy.core.fromnumeric import argsort
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
import matplotlib.pyplot as plt

def otsu_thresh(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def niblack_thresh(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    res = img.copy()
    thresh_niblack = threshold_niblack(img, window_size=25, k=0.8)
    thresh_niblack = cv2.ximgproc.niBlackThreshold(img, maxValue=255, type=cv2.THRESH_BINARY_INV, blockSize=2*11+1, k=-0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)

    binary_niblack = img > thresh_niblack

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if binary_niblack[i][j]==True:
                res[i][j]=255
            else:
                res[i][j]=0
    return res

def sauvola_thresh(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    res = img.copy()
    thresh_sauvola = threshold_sauvola(img, window_size=25)
    thresh_sauvola = cv2.ximgproc.niBlackThreshold(img, maxValue=255, type=cv2.THRESH_BINARY_INV, blockSize=2*11+1, k=-0.2, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)

    binary_sauvola = img > thresh_sauvola

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if binary_sauvola[i][j]==True:
                res[i][j]=255
            else:
                res[i][j]=0
    return res

def cbir():
    im1 = cv2.imread('static/images/000118 (5).png')
    im_resized1 = cv2.resize(im1, (225, 225), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized1, cv2.COLOR_BGR2RGB))
    plt.show()

    im2 = cv2.imread('static/images/000115 (4).png')
    im_resized2 = cv2.resize(im2, (225, 225), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized2, cv2.COLOR_BGR2RGB))
    plt.show()

    im3 = cv2.imread('static/images/000119 (6).png')
    im_resized3 = cv2.resize(im3, (225, 225), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized3, cv2.COLOR_BGR2RGB))
    plt.show()
    return 0