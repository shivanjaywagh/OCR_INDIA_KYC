import pytesseract
import cv2
import re
from newnow import intergral_thresh
img = cv2.imread("Amit-PAN_card2.jpg")

cv2.imshow("image.jpg", img)
cv2.waitKey(0) #image will not show until this is called
cv2.destroyWindow('image.jpg') #make sure window closes cleanly
img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = intergral_thresh(img_bw)

rot_data = pytesseract.image_to_osd(img);
print("[OSD] "+rot_data)
rot = re.search(r'(?<=Rotate: )\d+', rot_data).group(0)

angle = float(rot)
if angle > 0:
    angle = 360 - angle
print("[ANGLE] "+str(angle))

if (str(int(angle)) == '0'):
    rotated = img
    
elif (str(int(angle)) == '90'):
    rotated = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
elif (str(int(angle)) == '180'):
    rotated = cv2.rotate(img,cv2.ROTATE_180)
    
elif (str(int(angle)) == '270'):
    rotated = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite("orientation_out.jpg",rotated)
'''
cv2.imshow("image.jpg", rotated)
cv2.waitKey(0) #image will not show until this is called
cv2.destroyWindow('image.jpg') #make sure window closes cleanly
'''