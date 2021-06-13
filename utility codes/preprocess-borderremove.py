import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('nowwwwwww2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = np.zeros(image.shape, dtype=np.uint8)

'''
cv2.imshow('Original Image', gray)
cv2.waitKey(0)
cv2.destroyWindow('Original Image') #make sure window closes cleanly

cv2.imshow('Original Image mask', mask)
cv2.waitKey(0)
cv2.destroyWindow('Original Image mask') #make sure window closes cleanly
'''
# Draw contours onto a mask
cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# The cv2.fillPoly() function can be used to fill in any shape
cv2.fillPoly(mask, cnts, [255,255,255])

#cv2.imshow('FillPoly mask', mask)
#cv2.waitKey(0)
#cv2.destroyWindow('FillPoly mask') #make sure window closes cleanly

#Invert mask
mask = 255 - mask

#cv2.imshow('Inverted Mask', mask)
#cv2.waitKey(0)
#cv2.destroyWindow('Inverted Mask') #make sure window closes cleanly

#Bitwise-or with original image, in this we merge both mask and original image by applying OR operation
result = cv2.bitwise_or(image, mask)

#cv2.imshow('Again original Image', gray)
#cv2.waitKey(0)
#cv2.destroyWindow('Again original Image') #make sure window closes cleanly

#cv2.imshow('Bitwise-or with original Image & mask without borders', result)
#cv2.waitKey(0)
#cv2.destroyWindow('Bitwise-or with original Image & mask without borders') #make sure window closes cleanly
cv2.imwrite("prepro.jpg", result)