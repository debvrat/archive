import numpy as np
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def applyFilter(image, kernel_type,kernel_x,kernel_y):
	print('Applying ' + kernel_type + ' filter ...')

	kernel_dim = kernel_x.shape[0]
	G_x = np.zeros_like(image)
	G_y = np.zeros_like(image)

	row, col = image.shape
	image_padded = np.zeros((row + 2, col + 2))   
	image_padded[1:-1, 1:-1] = image

	for i in range(col):
		for j in range(row):
			G_x[j,i]=(kernel_x*image_padded[j:j+kernel_dim,i:i+kernel_dim]).sum()
			G_y[j,i]=(kernel_y*image_padded[j:j+kernel_dim,i:i+kernel_dim]).sum()

	gradient_l2 = (G_x**2 + G_y**2)**0.5
	gradient_l1 = abs(G_x) + abs(G_y) 

	return gradient_l2, gradient_l1

def applyCanny(image, minVal, maxVal):
	print('Processing Canny filter with minVal = ' + str(minVal) + ' and maxVal = ' + str(maxVal)) 
	image_canny = cv2.Canny(image,minVal,maxVal)
	return image_canny

def main():
	image = cv2.imread('../input_data/barbara.jpg')
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = exposure.equalize_adapthist(image_gray/np.max(np.abs(image_gray)), clip_limit=0.03)

	roberts_x = [[0,1],[-1,0]]
	roberts_y = [[1,0],[0,-1]]
	prewitt_x = [[-1,0,1],[-1,0,1],[-1,0,1]]
	prewitt_y = [[1,1,1],[0,0,0],[-1,-1,-1]]
	sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
	sobel_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
	laplace_x = [[0,1,0],[1,-4,1],[0,1,0]]
	laplace_y = [[1,1,1],[1,-8,1],[1,1,1]]

	print(' ')
	print('### Q1.2&3 Edge Detection ###')
	print(' ')
	print('Choose a filter type:')
	print('   1. Roberts')
	print('   2. Prewitt')
	print('   3. Sobel')
	print('   4. Laplace')

	inp = input()
	kernel_type = ''
	if (inp == '1'):
		kernel_x = np.vstack(roberts_x)
		kernel_y = np.vstack(roberts_x)
		kernel_type = 'roberts'
	elif (inp == '2'):
		kernel_x = np.vstack(prewitt_x)
		kernel_y = np.vstack(prewitt_y)
		kernel_type = 'prewitt'
	elif (inp == '3'):
		kernel_x = np.vstack(sobel_x)
		kernel_y = np.vstack(sobel_y)
		kernel_type = 'sobel'
	elif (inp == '4'):
		kernel_x = np.vstack(laplace_x)
		kernel_y = np.vstack(laplace_y)
		kernel_type = 'laplace'
	else:
		print('Enter a valid key')
		return
		
	l2, l1 = applyFilter(image, kernel_type, kernel_x, kernel_y)
	plt.imshow(l2, cmap='gray')
	plt.axis('off')
	plt.savefig('../output_data/1.2_'+kernel_type+'_L2.jpg', bbox_inches='tight')
	plt.imshow(l1, cmap='gray')
	plt.axis('off')
	plt.savefig('../output_data/1.2_'+kernel_type+'_L1.jpg', bbox_inches='tight')
	print(kernel_type + ' filter applied on Barbara.jpg') 

	minVal = 70
	maxVal = 100
	image_canny = applyCanny(image_gray, minVal, maxVal)
	plt.imshow(image_canny, cmap='gray')
	plt.axis('off')
	plt.savefig('../output_data/canny.jpg', bbox_inches='tight')
	print('Canny filter applied on Barabara.jpg')

	row,col = image.shape
	mean = 0.5
	var = 0.01
	sigma = var**0.5
	gauss = np.random.normal(mean,sigma,(row,col))
	gauss = gauss.reshape(row,col)
	noisy = image + gauss
	plt.imshow(noisy, cmap='gray')
	plt.axis('off')
	plt.savefig('../output_data/noisy.png', bbox_inches='tight')
	print('Gaussian noise added to Barbara.jpg')

	l2_noisy, l1_noisy = applyFilter(noisy, kernel_type, kernel_x, kernel_y)
	plt.imshow(l2_noisy, cmap='gray')
	plt.axis('off')
	plt.savefig('../output_data/noisyFiltered.jpg', bbox_inches='tight')
	print(kernel_type + ' filter applied on noisy Barbara.jpg') 


	print('Done')

	return

if __name__ == "__main__":
    main()