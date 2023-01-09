import cv2
import numpy as np
import matplotlib.pyplot as plt


#image up sampling
def upsample(img, scale):
    h, w = img.shape[:2]
    img = cv2.resize(img, (w*scale, h*scale))
    return img

#image down sampling
def downsample(img, scale):
    h, w = img.shape[:2]
    img = cv2.resize(img, (w/scale, h/scale), interpolation=cv2.INTER_CUBIC)
    return img

#image sharpening
def sharpen(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    return img

#image blurring
def blur(img):
    img = cv2.blur(img, (3,3))
    return img

#image edge detection
def edge(img):
    img = cv2.Canny(img, 100, 200)
    return img

#image thresholding
def threshold(img):
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img

#image dilation
def dilate(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations = 1)
    return img

#image erosion
def erode(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.erode(img, kernel, iterations = 1)
    return img

#image opening
def opening(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img

#image closing
def closing(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

#image gradient
def gradient(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return img

#image blackhat
def blackhat(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return img

#image tophat
def tophat(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    return img

#image skeletonization
def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel

#image contouring
def contour(img):
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img


scale=3
img=cv2.imread('sword.png',0)
plt.figure(1)
plt.subplot(121)
plt.imshow(img)
img2=upsample(img, scale)
plt.subplot(122)
plt.imshow(img2)
plt.show()