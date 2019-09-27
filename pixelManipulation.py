#python 3.7.4
from PIL import Image
import numpy as np 
import cv2 
from matplotlib import pyplot as plt 

def mostFreqClr (img):
	w, h = img.size
	colourMap = img.getcolors(w*h)
	frequent_colour = colourMap[0] #initialize
	for count, colour in colourMap:
		if count > frequent_colour[0]:
			frequent_colour = (count, colour)

	return frequent_colour

def mergeImage(fg, bg):
	mask = np.zeros(fg.shape[:2], np.uint8) 
	backgroundModel = np.zeros((1, 65), np.float64) 
	foregroundModel = np.zeros((1, 65), np.float64) 
	rectangle = (500, 50, 230, 700) 
	cv2.grabCut(fg, mask, rectangle,   
	            backgroundModel, foregroundModel, 
	            3, cv2.GC_INIT_WITH_RECT) 
	   
	mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
	   
	fg = fg * mask2[:, :, np.newaxis] 
	  
	img = cv2.add(fg, bg) 
	
	cv2.imwrite('../output_data/q1_merged.jpg',img)  
	cv2.imshow('merged', img) 
	cv2.waitKey(0) 
	

def main():
	print ('### most frequently occuring color ###')
	print ('enter file name')
	file1 = input()
	img = Image.open("../input_data/" + str(file1))
	count, rgb = mostFreqClr(img)
	print ('Colour(rgb) ' + str(rgb) + ' occurs ' + str(count) + ' times')
	img.close()

	print ('### merge image ###')
	print ('enter foreground image')
	file2 = input()
	fg = cv2.imread("../input_data/" + str(file2))
	print ('enter background image')
	file3 = input()
	bg = cv2.imread("../input_data/" + str(file3))
	mergeImage (fg, bg)


if __name__ == "__main__":
    main()
