# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:55:04 2020

@author: Venom
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import utils
import CannyOperator as ced
from scipy import signal as sig
from scipy import ndimage as ndi
from ImageSegmentation import otsu

# Fungsi untuk membaca citra
def inputCitra(im):
    hasil = plt.imread(im)
    return(hasil)

# Fungsi untuk mengubah citra ke Grayscale
def rgbToGray(im):
    hasil = np.int16(0.299*im[:,:,0]+0.587*im[:,:,1]+(1-0.299-0.587)*im[:,:,2])
    return(hasil)

def gradient_x(imggray):
    ##Sobel operator kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return sig.convolve2d(imggray, kernel_x, mode='same')

def gradient_y(imggray):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(imggray, kernel_y, mode='same')

def detM(Ixx,Iyy,Ixy):
    return Ixx*Iyy-Ixy**2

def traceM(Ixx,Iyy):
    return Ixx+Iyy

def crf(Ixx, Iyy, Ixy, e):
    return detM(Ixx,Iyy,Ixy)/(traceM(Ixx,Iyy)**2+e)    

def main():
    imasli = plt.imread("images/3.jpg")
    imgs = utils.load_data()    
    utils.visualize(imgs, 'gray')
    detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.4, highthreshold=0.4, weak_pixel=100)
    imgs_final = detector.detect()
    utils.visualize(imgs_final, 'gray')
    imCanny = imgs_final[0]
    plt.subplot(111); plt.imshow(imCanny, cmap='gray'); plt.title('final')
    plt.show()
    I_x = gradient_x(imCanny)
    I_y = gradient_y(imCanny)
    Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma=1)
    response = crf(Ixx, Iyy, Ixy, 0.000001)
    imOutput = otsu(response)
    print(response)
    print(imOutput.size)
    plt.subplot(111); plt.imshow(imOutput, cmap='gray'); plt.title('final')
    plt.show()
    img_copy_for_corners = np.copy(imasli)
    img_copy_for_edges = np.copy(imasli)
    for rowindex, res in enumerate(response):
        for colindex, r in enumerate(res):
            if r > 0:
                # this is a corner
                img_copy_for_corners[rowindex, colindex] = [255,0,0]
            elif r < 0:
                # this is an edge
                img_copy_for_edges[rowindex, colindex] = [0,255,0]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    ax[0].set_title("corners found")
    ax[0].imshow(img_copy_for_corners)
    ax[1].set_title("edges found")
    ax[1].imshow(img_copy_for_edges)
    plt.show()
main()
    
    
