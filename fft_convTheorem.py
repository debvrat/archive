print('start')
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from scipy import signal

def convolve(image, kernel):
    
    out = np.zeros_like(image)
    image = (image-np.min(image))/(np.max(image)-np.min(image))

    row, col = image.shape
    k_row, k_col = kernel.shape
    pad_row = k_row-1
    pad_col = k_col-1
    
    image_padded = np.zeros((row + pad_row, col + pad_col))  
    
    img_rIdx = int(pad_row/2)
    img_cIdx = int(pad_col/2)
    
    image_padded[img_rIdx:-img_rIdx, img_cIdx:-img_cIdx] = image
    for i in range(col):
        for j in range(row):
            out[j,i]=(kernel*image_padded[j:j+k_row,i:i+k_col]).sum()
    return np.round(out*255)

img = cv2.imread('../input_data/lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('../input_data/bricks.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

convolved = convolve(img, img2)
cv2.imwrite('convolve.jpg', convolved)
print('convolution complete')

img2pad = np.pad(img2, ((90,89),(50,49)), mode='constant')

img_f = np.fft.fft2(img)
img2pad_f = np.fft.fft2(img2pad)
convolved_f = np.fft.fft2(convolved)
cv2.imwrite('convolve-fft-abs.jpg', abs(convolved_f))
product = np.multiply(img_f, img2pad_f)
cv2.imwrite('product-abs.jpg', abs(product))

def calcConvTime(img1, img2): #grayscale image matrices
    start = time.time()
    out = signal.convolve2d(img1, img2)
    end = round(time.time() - start, 2)
    return end

def calcDFTtime(img1z, img2z): #appropriately zero padded images
    start = time.time()
    
    img1_f = np.fft.fft2(img1z)
    img2_f = np.fft.fft2(img2z)
    prod = np.multiply(img1_f, img2_f)
    idft = np.fft.ifft2(prod)

    end = round(time.time() - start, 2)
    return end

print('Convolution time:')
print(calcConvTime(img, img2))
print('IDFT time:')
print(calcDFTtime(img, img2pad))
