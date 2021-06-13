import cv2
import numpy as np
import glob
import random

import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
import re
from langdetect import detect_langs
from PIL import ImageGrab
import time
import numpy as np
from matplotlib import pyplot as plt
from skimage import data
from skimage.filters import threshold_multiotsu
import tempfile
IMAGE_SIZE = 1800
from matplotlib import pyplot as plt

exit = 0
def set_image_dpi(file_path):
	im = Image.open(file_path)
	length_x, width_y = im.size
	factor = max(1, int(IMAGE_SIZE / length_x))
	size = factor * length_x, factor * width_y
	# size = (1800, 1800)
	im_resized = im.resize(size, Image.ANTIALIAS)
	temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
	temp_filename = temp_file.name
	im_resized.save(temp_filename, dpi=(300, 300))
	return temp_filename


def remove_noise_and_smooth(file_name):
	#img = set_image_dpi(file_name)
	img = cv2.imread(file_name, 0)
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("grayyyy.jpg", img)
	histr = cv2.calcHist([img],[0],None,[256],[0,256])
	# show the plotting graph of an image
	min_val = min(histr)
	plt.plot(histr)
	plt.show()
	cv2.imshow("img", img)
	cv2.waitKey(0)
	th, threshed = cv2.threshold(img, 158, 255, cv2.THRESH_BINARY)
	cv2.imwrite("grayyyythresh150.jpg", threshed)
	print(histr)
	print(len(histr))
	

	blur = cv2.GaussianBlur(img,(5,5),0)
	ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_OTSU+cv2.THRESH_OTSU)
	cv2.imwrite("grayyyyotsu.jpg", th3)
	

	filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	cv2.imwrite("adaptThresh.jpg", filtered)

	blur = cv2.GaussianBlur(filtered,(5,5),0)
	ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
	median = cv2.medianBlur(th3,5)
	median = cv2.medianBlur(median,5)
	median = cv2.medianBlur(median,5)
	median = cv2.medianBlur(median,5)
	# find minima in index 100 to 180


	cv2.imshow("grayyyy.jpg", median)
	histr = cv2.calcHist([median],[0],None,[256],[0,256])
	# show the plotting graph of an image
	min_val = min(histr)
	plt.plot(histr)
	plt.show()






	
	median = cv2.bitwise_not(median)
	kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
	img_dil = cv2.dilate(median, kernel, iterations = 1)
	median = cv2.bitwise_not(img_dil)


	#kernel = np.ones((1, 1), np.uint8)
	#opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
	#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	#img = image_smoothening(img)
	#or_image = cv2.bitwise_or(img, closing)
	# plot all the images and their histograms
	#thresholds = threshold_multiotsu(img, classes = 20)
	'''
	images = [blur, 0, thresholds]
	titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
    	      'Original Noisy Image','Histogram',"Otsu's Thresholding",
        	  'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

	for i in range(1):
		plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
		
		plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
		
		plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
		
	plt.show()
	'''

	return median


#imgg = remove_noise_and_smooth('D:/fortiate/project/a-copy.jpg')
#imgg = remove_noise_and_smooth('D:/fortiate/project/IMG_20210415_163829.jpg')
#imgg = remove_noise_and_smooth('D:/fortiate/project/sample_photos/pancard_type1.jpg')
#imgg = remove_noise_and_smooth('D:/fortiate/project/sample_photos/pancard_type2.jpg')
#imgg = remove_noise_and_smooth('D:/fortiate/project/sample_photos/aa.jpg')
imgg = remove_noise_and_smooth('D:/fortiate/project/sample_photos/aaa.jpg')






cv2.imwrite("smooth1otsu234after3.jpg", imgg)
#cv2.imwrite("smooth1otsu234before.jpg", imgg2)

