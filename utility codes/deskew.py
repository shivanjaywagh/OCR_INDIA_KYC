import cv2
import numpy as np
from newnow import intergral_thresh


image = cv2.imread('IMG_20210427_143530.jpg')

cv2.imshow("image.jpg", image)
cv2.waitKey(0) #image will not show until this is called
cv2.destroyWindow('image.jpg') #make sure window closes cleanly

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.bitwise_not(gray)
# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
#thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = intergral_thresh(gray)
# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
   angle = -(90 + angle)
# otherwise, just take the inverse of the angle to make
# it positive
else:
   angle = -angle

print("angle", angle)

shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)
height = image.shape[0]
width = image.shape[1]
center=(height/2, width/2)
matrix = cv2.getRotationMatrix2D( center=center, angle=angle, scale=1 )
image1 = cv2.warpAffine( src=image, M=matrix, dsize=shape )
cv2.imwrite('deskew_image_out.jpg', image1)

cv2.imshow("image.jpg", image1)
cv2.waitKey(0) #image will not show until this is called
cv2.destroyWindow('image.jpg') #make sure window closes cleanly
cv2.imwrite("deskewres.jpg", image1)

