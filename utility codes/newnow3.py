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
from PIL import Image 
import PIL 
exit = 0

import warnings
warnings.filterwarnings("ignore")



def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect




















def preocr(threshedfile):
	str1 = "start  "

	from pytesseract import Output
	d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT)
	print(d)
	n_boxes = len(d['level'])
	for i in range(n_boxes):
		if(d['text'][i] != "" and float(d['conf'][i] )>= 45.0 and d['text'][i] != " "):
			(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
			print(d['text'][i])
			print(d['conf'][i])
			print("******************************")
			str1 = str1 +"\n"+ d['text'][i]
			cv2.rectangle(threshedfile, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imwrite('result.png', threshedfile)
	str1 = str1.replace("|", "")
	str1 = str1.replace(".", "")
	str1 = str1.replace(",", " ")
	str1 = str1.replace("asta", " ")
	str1 = str1.replace("fara", " ")
	str1 = str1.replace("are", " ")
	str1 = str1.replace("Father", " ")
	str1 = str1.replace("Name", " ")
	str1 = str1.replace("NAME", " ")
	str1 = str1.replace("Father's", " ")
	str1 = str1.replace("Fathers", " ")
	str1 = str1.replace("FATHER", " ")
	str1 = str1.replace("FATHER'S", " ")
	str1 = str1.replace("FATHERS", " ")
	str1 = str1.replace("Birth", " ")
	str1 = str1.replace("BIRTH", " ")
	str1 = str1.replace("Year", " ")
	str1 = str1.replace("YEAR", " ")
	str1 = str1.replace("India", " ")
	str1 = str1.replace("INDIA", " ")
	str1 = str1.replace("OF", " ")
	str1 = str1.replace("Of", " ")
	str1 = str1.replace("of", " ")
	str1 = str1.replace("GOVERNMENT", " ")
	str1 = str1.replace("Government", " ")
	str1 = str1.replace("GOVT", " ")
	str1 = str1.replace("Govt", " ")
	str1 = str1.replace("Permanent", " ")
	str1 = str1.replace("PERMANENT", " ")
	str1 = str1.replace("PARMANENT", " ")
	str1 = str1.replace("Parmanent", " ")
	str1 = str1.replace("Account", " ")
	str1 = str1.replace("ACCOUNT", " ")

	str1 = str1.replace("Number", " ")
	str1 = str1.replace("NUMBER", " ")
	str1 = str1.replace("Card", " ")
	str1 = str1.replace("CARD", " ")
	str1 = str1.replace("Signature", " ")
	str1 = str1.replace("SIGNATURE", " ")
	str1 = str1.replace("DATE", " ")
	str1 = str1.replace("Date", " ")
	str1 = str1.replace("INCOME", " ")
	str1 = str1.replace("Income", " ")

















	str1 = str1.replace(" ", "")













	print(str1)
	return str1


def preocr_name(card,threshedfile):
	str1 = "start  "
	card = card
	name_list = []
	conf_list = []
	
	new_dict = {}
	if card==1:     #pan1
		height = threshedfile.shape[0]
		width = threshedfile.shape[1]

		#int(0.011*width)<left<int(0.426*width)
		#int(0.240*height)<top<int(0.507*height)




		from pytesseract import Output
		#d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT) #config = r'-l eng+hin --psm 6'
		d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT , config = r'-l eng --psm 6') #config = r'-l eng+hin --psm 6'

		print(d)
		n_boxes = len(d['level'])
		for i in range(n_boxes):
			if(d['text'][i] != "" and float(d['conf'][i] )>= 45.0 and d['text'][i] != " "):
				if (int(0.011*width)<int(d['left'][i]) and int(d['left'][i])<int(0.426*width)) and (int(0.240*height)<int(d['top'][i]) and int(d['top'][i])<int(0.507*height)):
					(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
					print(d['text'][i])
					print(d['conf'][i])
					print("******************************")

					#name_list = name_list + "\n" + d['text'][i]
					#conf_list = conf_list + "\n" + d['conf'][i]
					#dict_list[i][0] = d['text'][i]
					#dict_list[i][1] = d['conf'][i]
					new_dict[str(d['text'][i])] = d['conf'][i]

					


					str1 = str1 +"\n"+ d['text'][i]
					cv2.rectangle(threshedfile, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imwrite('resultname.png', threshedfile)
		str1 = str1.replace("|", "")
		str1 = str1.replace(".", "")
		str1 = str1.replace(",", " ")
		str1 = str1.replace("asta", " ")
		str1 = str1.replace("fara", " ")
		str1 = str1.replace("are", " ")
		str1 = str1.replace("Father", " ")
		str1 = str1.replace("Name", " ")
		str1 = str1.replace("NAME", " ")
		str1 = str1.replace("Father's", " ")
		str1 = str1.replace("Fathers", " ")
		str1 = str1.replace("FATHER", " ")
		str1 = str1.replace("FATHER'S", " ")
		str1 = str1.replace("FATHERS", " ")
		str1 = str1.replace("Birth", " ")
		str1 = str1.replace("BIRTH", " ")
		str1 = str1.replace("Year", " ")
		str1 = str1.replace("YEAR", " ")
		str1 = str1.replace("India", " ")
		str1 = str1.replace("INDIA", " ")
		str1 = str1.replace("OF", " ")
		str1 = str1.replace("Of", " ")
		str1 = str1.replace("of", " ")
		str1 = str1.replace("GOVERNMENT", " ")
		str1 = str1.replace("Government", " ")
		str1 = str1.replace("GOVT", " ")
		str1 = str1.replace("Govt", " ")
		str1 = str1.replace("Permanent", " ")
		str1 = str1.replace("PERMANENT", " ")
		str1 = str1.replace("PARMANENT", " ")
		str1 = str1.replace("Parmanent", " ")
		str1 = str1.replace("Account", " ")
		str1 = str1.replace("ACCOUNT", " ")

		str1 = str1.replace("Number", " ")
		str1 = str1.replace("NUMBER", " ")
		str1 = str1.replace("Card", " ")
		str1 = str1.replace("CARD", " ")
		str1 = str1.replace("Signature", " ")
		str1 = str1.replace("SIGNATURE", " ")
		str1 = str1.replace("DATE", " ")
		str1 = str1.replace("Date", " ")
		str1 = str1.replace("INCOME", " ")
		str1 = str1.replace("Income", " ")





		str1 = str1.replace(" ", "")

		print("preocr_name -----")
		print(str1)
		print(new_dict)
		return str1





	if card==2:     #pan
		height = threshedfile.shape[0]
		width = threshedfile.shape[1]

		#int(0.011*width)<left<int(0.426*width)
		#int(0.240*height)<top<int(0.507*height)




		from pytesseract import Output
		#d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT) #config = r'-l eng+hin --psm 6'
		d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT , config = r'-l eng --psm 6') #config = r'-l eng+hin --psm 6'

		print(d)
		n_boxes = len(d['level'])
		for i in range(n_boxes):
			if(d['text'][i] != "" and float(d['conf'][i] )>= 45.0 and d['text'][i] != " "):
				if (int(0.027*width)<int(d['left'][i]) and int(d['left'][i])<int(0.60*width)) and (int(0.502*height)<int(d['top'][i]) and int(d['top'][i])<int(0.640*height)):
					(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
					print(d['text'][i])
					print(d['conf'][i])
					print("******************************")
					new_dict[str(d['text'][i])] = d['conf'][i]


					str1 = str1 +"\n"+ d['text'][i]
					cv2.rectangle(threshedfile, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imwrite('resultname.png', threshedfile)
		str1 = str1.replace("|", "")
		str1 = str1.replace(".", "")
		str1 = str1.replace(",", " ")
		str1 = str1.replace("asta", " ")
		str1 = str1.replace("fara", " ")
		str1 = str1.replace("are", " ")
		str1 = str1.replace("Father", " ")
		str1 = str1.replace("Name", " ")
		str1 = str1.replace("NAME", " ")
		str1 = str1.replace("Father's", " ")
		str1 = str1.replace("Fathers", " ")
		str1 = str1.replace("FATHER", " ")
		str1 = str1.replace("FATHER'S", " ")
		str1 = str1.replace("FATHERS", " ")
		str1 = str1.replace("Birth", " ")
		str1 = str1.replace("BIRTH", " ")
		str1 = str1.replace("Year", " ")
		str1 = str1.replace("YEAR", " ")
		str1 = str1.replace("India", " ")
		str1 = str1.replace("INDIA", " ")
		str1 = str1.replace("OF", " ")
		str1 = str1.replace("Of", " ")
		str1 = str1.replace("of", " ")
		str1 = str1.replace("GOVERNMENT", " ")
		str1 = str1.replace("Government", " ")
		str1 = str1.replace("GOVT", " ")
		str1 = str1.replace("Govt", " ")
		str1 = str1.replace("Permanent", " ")
		str1 = str1.replace("PERMANENT", " ")
		str1 = str1.replace("PARMANENT", " ")
		str1 = str1.replace("Parmanent", " ")
		str1 = str1.replace("Account", " ")
		str1 = str1.replace("ACCOUNT", " ")

		str1 = str1.replace("Number", " ")
		str1 = str1.replace("NUMBER", " ")
		str1 = str1.replace("Card", " ")
		str1 = str1.replace("CARD", " ")
		str1 = str1.replace("Signature", " ")
		str1 = str1.replace("SIGNATURE", " ")
		str1 = str1.replace("DATE", " ")
		str1 = str1.replace("Date", " ")
		str1 = str1.replace("INCOME", " ")
		str1 = str1.replace("Income", " ")


		str1 = str1.replace(" ", "")












		print("preocr_name -----")
		print(str1)	
		print(new_dict)

		return str1



	if card==3:     #aadhar
		height = threshedfile.shape[0]
		width = threshedfile.shape[1]

		#int(0.011*width)<left<int(0.426*width)
		#int(0.240*height)<top<int(0.507*height)




		from pytesseract import Output
		#d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT) #config = r'-l eng+hin --psm 6'
		d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT , config = r'-l eng --psm 6') #config = r'-l eng+hin --psm 6'

		print(d)
		n_boxes = len(d['level'])
		for i in range(n_boxes):
			if(d['text'][i] != "" and float(d['conf'][i] )>= 45.0 and d['text'][i] != " "):
				if (int(0.274*width)<int(d['left'][i]) and int(d['left'][i])<int(0.8226*width)) and (int(0.258*height)<int(d['top'][i]) and int(d['top'][i])<int(0.416*height)):
					(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
					print(d['text'][i])
					print(d['conf'][i])
					print("******************************")
					new_dict[str(d['text'][i])] = d['conf'][i]
					

					str1 = str1 +"\n"+ d['text'][i]
					cv2.rectangle(threshedfile, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imwrite('resultname.png', threshedfile)
		str1 = str1.replace("|", "")
		str1 = str1.replace(".", "")
		str1 = str1.replace(",", " ")
		str1 = str1.replace("asta", " ")
		str1 = str1.replace("fara", " ")
		str1 = str1.replace("are", " ")
		str1 = str1.replace("Father", " ")
		str1 = str1.replace("Name", " ")
		str1 = str1.replace("NAME", " ")
		str1 = str1.replace("Father's", " ")
		str1 = str1.replace("Fathers", " ")
		str1 = str1.replace("FATHER", " ")
		str1 = str1.replace("FATHER'S", " ")
		str1 = str1.replace("FATHERS", " ")
		str1 = str1.replace("Birth", " ")
		str1 = str1.replace("BIRTH", " ")
		str1 = str1.replace("Year", " ")
		str1 = str1.replace("YEAR", " ")
		str1 = str1.replace("India", " ")
		str1 = str1.replace("INDIA", " ")
		str1 = str1.replace("OF", " ")
		str1 = str1.replace("Of", " ")
		str1 = str1.replace("of", " ")
		str1 = str1.replace("GOVERNMENT", " ")
		str1 = str1.replace("Government", " ")
		str1 = str1.replace("GOVT", " ")
		str1 = str1.replace("Govt", " ")
		str1 = str1.replace("Permanent", " ")
		str1 = str1.replace("PERMANENT", " ")
		str1 = str1.replace("PARMANENT", " ")
		str1 = str1.replace("Parmanent", " ")
		str1 = str1.replace("Account", " ")
		str1 = str1.replace("ACCOUNT", " ")

		str1 = str1.replace("Number", " ")
		str1 = str1.replace("NUMBER", " ")
		str1 = str1.replace("Card", " ")
		str1 = str1.replace("CARD", " ")
		str1 = str1.replace("Signature", " ")
		str1 = str1.replace("SIGNATURE", " ")
		str1 = str1.replace("DATE", " ")
		str1 = str1.replace("Date", " ")
		str1 = str1.replace("INCOME", " ")
		str1 = str1.replace("Income", " ")


		str1 = str1.replace(" ", "")












		print("preocr_name -----")
		print(str1)	
		print(new_dict)

		return str1


	if card==4:     #panoldest
		height = threshedfile.shape[0]
		width = threshedfile.shape[1]

		#int(0.011*width)<left<int(0.426*width)
		#int(0.240*height)<top<int(0.507*height)




		from pytesseract import Output
		#d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT) #config = r'-l eng+hin --psm 6'
		d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT , config = r'-l eng --psm 6') #config = r'-l eng+hin --psm 6'

		print(d)
		n_boxes = len(d['level'])
		for i in range(n_boxes):
			if(d['text'][i] != "" and float(d['conf'][i] )>= 45.0 and d['text'][i] != " "):
				if (int(0.027*width)<int(d['left'][i]) and int(d['left'][i])<int(0.8226*width)) and (int(0.150*height)<int(d['top'][i]) and int(d['top'][i])<int(0.416*height)):
					(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
					print(d['text'][i])
					print(d['conf'][i])
					print("******************************")
					new_dict[str(d['text'][i])] = d['conf'][i]

					str1 = str1 +"\n"+ d['text'][i]
					cv2.rectangle(threshedfile, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imwrite('resultname.png', threshedfile)
		str1 = str1.replace("|", "")
		str1 = str1.replace(".", "")
		str1 = str1.replace(",", " ")
		str1 = str1.replace("asta", " ")
		str1 = str1.replace("fara", " ")
		str1 = str1.replace("are", " ")
		str1 = str1.replace("Father", " ")
		str1 = str1.replace("Name", " ")
		str1 = str1.replace("NAME", " ")
		str1 = str1.replace("Father's", " ")
		str1 = str1.replace("Fathers", " ")
		str1 = str1.replace("FATHER", " ")
		str1 = str1.replace("FATHER'S", " ")
		str1 = str1.replace("FATHERS", " ")
		str1 = str1.replace("Birth", " ")
		str1 = str1.replace("BIRTH", " ")
		str1 = str1.replace("Year", " ")
		str1 = str1.replace("YEAR", " ")
		str1 = str1.replace("India", " ")
		str1 = str1.replace("INDIA", " ")
		str1 = str1.replace("OF", " ")
		str1 = str1.replace("Of", " ")
		str1 = str1.replace("of", " ")
		str1 = str1.replace("GOVERNMENT", " ")
		str1 = str1.replace("Government", " ")
		str1 = str1.replace("GOVT", " ")
		str1 = str1.replace("Govt", " ")
		str1 = str1.replace("Permanent", " ")
		str1 = str1.replace("PERMANENT", " ")
		str1 = str1.replace("PARMANENT", " ")
		str1 = str1.replace("Parmanent", " ")
		str1 = str1.replace("Account", " ")
		str1 = str1.replace("ACCOUNT", " ")

		str1 = str1.replace("Number", " ")
		str1 = str1.replace("NUMBER", " ")
		str1 = str1.replace("Card", " ")
		str1 = str1.replace("CARD", " ")
		str1 = str1.replace("Signature", " ")
		str1 = str1.replace("SIGNATURE", " ")
		str1 = str1.replace("DATE", " ")
		str1 = str1.replace("Date", " ")
		str1 = str1.replace("INCOME", " ")
		str1 = str1.replace("Income", " ")


		str1 = str1.replace(" ", "")












		print("preocr_name -----")
		print(str1)	
		print(new_dict)

		return str1





























def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped































def crop_by_edges(img):
	#img_org = cv2.imread(images_path)
	
	img_org = img
	#ratio = image_org.shape[0] / 500.0
	ratio = 1
	
	#print(type(img))
	size = np.shape(img_org)
	img_org2 = img_org.copy()
	#img_bw = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

	ret3,img_thr = cv2.threshold(img_org,90,255,cv2.THRESH_TOZERO)
	#th3 = cv2.adaptiveThreshold(img_thr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	
	#Canny's Edge Detection
	img_edg  = cv2.Canny(img_thr,10,50)    #earlier it was img_thr
	#increasing the thickness of the edges
	kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (7, 7))
	img_dil = cv2.dilate(img_edg, kernel, iterations = 1)
	#finding the contours and sorting them in descending order
	(contours ,hierarchye) = cv2.findContours(img_dil.copy(), 1, 2)
	cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
	#giving shape to the contours
	screenCnt = []
	for c in cnts:
		peri = cv2.arcLength(c, True) #true=closed contour
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)#approx PolyDP shapes it approximately to a nearby image size
		# if our approximated contour has four points, then
		# we can assume that we have found our screen
		xx,yy,ww,hh = cv2.boundingRect(c)
		aspect_ratio = float(ww)/hh
		height, width = img_org.shape[:2]
		areaa = height * width
		flag = areaa/cv2.contourArea(c)
		#if (len(approx) == 4 and flag<2.5) and (aspect_ratio<1.8 and aspect_ratio>1.2):
		if (len(approx) == 4 and (aspect_ratio<1.8 and aspect_ratio>1.2) and flag<4):
			screenCnt = approx
			print("screenCNt below:")
			print(screenCnt)
			break
	if screenCnt == []:
		print("no contours")
		# deskew
		return img_org




	warped = four_point_transform(img_org, screenCnt.reshape(4, 2) * ratio)



	cv2.imwrite("warped.jpg",warped)
	#drawing contour around card plate
	mask = np.zeros(img_org.shape, dtype=np.uint8)
	roi_corners = np.array(screenCnt ,dtype=np.int32)
	ignore_mask_color = (255,)*1
	cv2.fillPoly(mask, roi_corners , ignore_mask_color)
	cv2.drawContours(img_org, [screenCnt], -40, (100, 255, 100), 9)
	cv2.imwrite("nowwwwwwwedgeddrawn.jpg", img_org)

	ys =[screenCnt[0,0,1] , screenCnt[1,0,1] ,screenCnt[2,0,1] ,screenCnt[3,0,1]]
	xs =[screenCnt[0,0,0] , screenCnt[1,0,0] ,screenCnt[2,0,0] ,screenCnt[3,0,0]]
	ys_sorted_index = np.argsort(ys)
	xs_sorted_index = np.argsort(xs)
	x1 = screenCnt[xs_sorted_index[0],0,0]
	x2 = screenCnt[xs_sorted_index[3],0,0]
	y1 = screenCnt[ys_sorted_index[0],0,1]
	y2 = screenCnt[ys_sorted_index[3],0,1]
	img_plate = img_org2[y1:y2 , x1:x2]

	return img_plate





def find_card_type(imggg):
	# Load Yolo pan2
	card_type = 4
	
	net = cv2.dnn.readNet("yolov3_training_last_pan2new.weights", "yolov3_testing.cfg")
	# Name custom object
	classes = ["pancard_2"]
	pan2=0
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	img = cv2.resize(imggg, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)
	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	conf_list_2 = [0]
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			#print(class_id)    #added by me
			confidence = scores[class_id]
			if confidence > 0.3:
				#pan2 = 1       # 2 --> pan2
				card_type = 2
				conf_list_2.append(confidence)


	print("confidence oftype2", conf_list_2)

	# Load Yolo pan1
	net = cv2.dnn.readNet("yolov3_training_last_pan.weights", "yolov3_testing.cfg")
	# Name custom object
	classes = ["pancard_1"]
	pan1 = 0
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	'''
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	'''
	#img = cv2.imread(img_plate)
	img = cv2.resize(imggg, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)
	#print(outs)#added by me
	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	conf_list_1 = [0]
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			#print(class_id)    #added by me
			confidence = scores[class_id]
			if confidence > 0.3:
				# Object detected
				#pan1 = 1       # 1 --> pan1 
				card_type = 1
				conf_list_1.append(confidence)

	print("confidence oftype1", conf_list_1)





	# Load Yolo aadhar
	net = cv2.dnn.readNet("yolov3_training_last_aadhar.weights", "yolov3_testing.cfg")
	# Name custom object
	classes = ["aadhar"]
	aadhar_flag = 0
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	'''
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	'''
	#img = cv2.imread(img_plate)
	img = cv2.resize(imggg, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)
	#print(outs)#added by me
	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	conf_list_3 = [0]
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			#print(class_id)    #added by me
			confidence = scores[class_id]
			if confidence > 0.3:
				# Object detected
				#aadhar_flag = 1       # 3 --> aadhar
				card_type = 3
				conf_list_3.append(confidence)
	
	print("confidence of type3", conf_list_3)

	if max(conf_list_3) > max(conf_list_2) and max(conf_list_3) > max(conf_list_1):
		card_type = 3
		return card_type

	elif max(conf_list_2) > max(conf_list_1) and max(conf_list_2) > max(conf_list_3):
		card_type = 2
		return card_type

	elif max(conf_list_1) > max(conf_list_2) and max(conf_list_1) > max(conf_list_3):
		card_type = 1
		return card_type




	return card_type   # 1 --> pan1 , 2--> pan 2 , 3 --> aadhar







def crop_out_header(img):
	pass







def intergral_thresh(imgg):
	ratio = 1
	T = 0.15
	#image = cv2.imdecode(np.fromfile('riyasat.jpg', dtype=np.uint8), 0)
	#image = cv2.imdecode(np.fromfile(img, )
	image = imgg
	img = cv2.resize(image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)), cv2.INTER_NEAREST)
	roii = cv2.integral(img)

	s = roii
	outputMat=np.zeros(img.shape, dtype=np.uint8)
	nRows = img.shape[0]
	nCols = img.shape[1]
	S = int(max(nRows, nCols) / 8)

	s2 = int(S / 4)

	for i in range(nRows):
		y1 = i - s2
		y2 = i + s2

		if (y1 < 0) :
			y1 = 0
		if (y2 >= nRows):
			y2 = nRows - 1

		for j in range(nCols):
			x1 = j - s2
			x2 = j + s2

			if (x1 < 0) :
				x1 = 0
			if (x2 >= nCols):
				x2 = nCols - 1
			count = (x2 - x1)*(y2 - y1)

			sum=s[y2][x2]-s[y2][x1]-s[y1][x2]+s[y1][x1]

			#if ((img[i][j] * count).astype(int) < (summ*(1.0 - T)).astype(int)):
			if ((int)(img[i][j] * count) < (int)(sum*(1.0 - T))):
				outputMat[i][j] = 255
		# print(i,j)
		# else:
		#outputMat[j][i] = 0
	outputMat = 255 - outputMat
	return outputMat




def ocr(card_type, threshed):
	result_dict = {}

	if card_type == 1:
		print("pancard1")
		
	
		
		
		#th, threshed = cv2.threshold(cropped, 127, 255, cv2.THRESH_TRUNC)
		#hImg, wImg,_ = img.shape
		## (3) Detect
		result = pytesseract.image_to_string((threshed), config = r'-l eng --psm 6')
		strr = preocr(threshed)
		str_name = preocr_name(card_type , threshed)

		female_pattern = r'(\bFEMALE\b|\bFemale\b|\bfemale\b|\b/FEMALE\b|\b/Female\b|\b/female\b)'
		search_result_female_pattern = re.findall(female_pattern , strr)
		result_dict["gender"] =search_result_female_pattern
		if bool(search_result_female_pattern):
			strr = strr.replace(str(search_result_female_pattern), " ")

		if not bool(search_result_female_pattern):
			male_pattern = r'(\bMALE\b|\bMale\b|\bmale\b|\b/MALE\b|\b/Male\b|\b/male\b)'
			search_result_male_pattern = re.findall(male_pattern , strr)
			result_dict["gender"] =search_result_male_pattern
			strr = strr.replace(str(search_result_male_pattern), " ")

		for word in result.split("\n"):
			if "”—" in word:
				word = word.replace("”—", ":")
  				
			#normalize NIK
			if "NIK" in word:
				nik_char = word.split()
			if "D" in word:
				word = word.replace("D", "0")
			if "?" in word:
				word = word.replace("?", "7") 
				
		#str_name = preocr_name(1,threshed)
		name_pattern = r'(\b[A-Z]{4,}\b|\b[A-Z]{1}[a-z]{2,}\b)'
		search_name_pattern = re.findall(name_pattern, str_name)
		search_name_pattern = search_name_pattern
		if not bool(search_name_pattern):
			print("namenotfound")
			name_pattern = r'(\b[A-Z]{4,}\b|\b[A-Z]{1}[a-z]{2,}\b)'
			search_name_pattern = re.findall(name_pattern, strr)
		
		result_dict["name"] = search_name_pattern[:3]
		
		date_pattern = r'([0-9]{2}\/[0-9]{2}\/[0-9]{4})'
		search_result_dob = re.findall(date_pattern, strr)    

		pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		search_result_pan_number_pattern = re.findall(pan_number_pattern, strr)

		result_dict["dob"] = search_result_dob
		result_dict["id_number"] = search_result_pan_number_pattern

		return result_dict

	if card_type == 2:
		print("pancard2")
		pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
		tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
		

		result = pytesseract.image_to_string((threshed), config = r'-l eng --psm 6')
		strr = preocr(threshed)
		str_name = preocr_name(card_type , threshed)

		female_pattern = r'(\bFEMALE\b|\bFemale\b|\bfemale\b|\b/FEMALE\b|\b/Female\b|\b/female\b)'
		search_result_female_pattern = re.findall(female_pattern , strr)
		result_dict["gender"] =search_result_female_pattern
		if bool(search_result_female_pattern):
			strr = strr.replace(str(search_result_female_pattern), " ")

		if not bool(search_result_female_pattern):
			male_pattern = r'(\bMALE\b|\bMale\b|\bmale\b|\b/MALE\b|\b/Male\b|\b/male\b)'
			search_result_male_pattern = re.findall(male_pattern , strr)
			result_dict["gender"] =search_result_male_pattern
			strr = strr.replace(str(search_result_male_pattern), " ")

		
		for word in result.split("\n"):
			if "”—" in word:
				word = word.replace("”—", ":")
  			#normalize NIK
			if "NIK" in word:
				nik_char = word.split()
			if "D" in word:
				word = word.replace("D", "0")
			if "?" in word:
				word = word.replace("?", "7") 

		pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		search_result_pan_number_pattern = re.findall(pan_number_pattern, strr)
		result_dict["id_number"] = search_result_pan_number_pattern 

		#strr = strr.replace(str(search_result_pan_number_pattern[:5]), " ")

		name_pattern = r'(\b[A-Z]{4,}\b|\b[A-Z]{1}[a-z]{2,}\b)'
		search_name_pattern = re.findall(name_pattern, str_name)
		search_name_pattern = search_name_pattern
		if not bool(search_name_pattern):
			print("namenotfound")
			name_pattern = r'(\b[A-Z]{4,}\b|\b[A-Z]{1}[a-z]{2,}\b)'
			search_name_pattern = re.findall(name_pattern, strr)


		result = pytesseract.image_to_string((threshed),config = r'-l eng --psm 6')

		for word in result.split("\n"):
			if "”—" in word:
				word = word.replace("”—", ":")
  			#normalize NIK
			if "NIK" in word:
				nik_char = word.split()
			if "D" in word:
				word = word.replace("D", "0")
			if "?" in word:
				word = word.replace("?", "7") 
		#print(result.split())
		#print(result[1])
		date_pattern = r'([0-9]{2}\/[0-9]{2}\/[0-9]{4})'
		search_result_dob = re.findall(date_pattern, strr)


		#print(search_result_pan_number_pattern) # Returns pan number
		#exit()
		result_dict["name"] = search_name_pattern[:3]
		result_dict["dob"] = search_result_dob


		return result_dict


	if card_type== 3:     #aadhar
		print("aadhar")
		pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
		tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
		
		
		## (3) Detect
		#result = pytesseract.image_to_string((threshed),config = r'-l eng+hin --psm 6')
		result = pytesseract.image_to_string((threshed),config = r'-l eng --psm 6')

		#print(result)
		
		for word in result.split("\n"):
			if "”—" in word:
				word = word.replace("”—", ":")
  			#normalize NIK
			if "NIK" in word:
				nik_char = word.split()
			if "D" in word:
				word = word.replace("D", "0")
			if "?" in word:
				word = word.replace("?", "7") 
		#print(result.split())
		#print(result[1])
		strr = preocr(threshed)
		str_name = preocr_name(card_type , threshed)

		date_pattern = r'([0-9]{2}\/[0-9]{2}\/[0-9]{4}|[0-9]{2}-[0-9]{2}-[0-9]{4})'
		search_result_dob = re.findall(date_pattern, strr)
		#print(search_result_dob) # Returns found object
		result_dict["dob"] = str(search_result_dob)
		if not bool(search_result_dob):
			dob_year_pattern = r'([0-9]{4})'
			dob_year = re.search(dob_year_pattern, strr)
			result_dict["dob"] = dob_year[0]

			#print(type(dob_year))
			#if bool(dob_year):
				#result_dict["dob"] = dob_year[0]
				#strr=strr.replace(str(dob_year[0])," ")

		female_pattern = r'(\bFEMALE\b|\bFemale\b|\bfemale\b|\b/FEMALE\b|\b/Female\b|\b/female\b)'
		search_result_female_pattern = re.findall(female_pattern , strr)
		result_dict["gender"] =search_result_female_pattern
		if bool(search_result_female_pattern):
			strr = strr.replace("FEMALE", " ")

		if not bool(search_result_female_pattern):
			male_pattern = r'(\bMALE\b|\bMale\b|\bmale\b|\b/MALE\b|\b/Male\b|\b/male\b)'
			search_result_male_pattern = re.findall(male_pattern , strr)
			result_dict["gender"] =search_result_male_pattern
			strr = strr.replace("MALE", " ")

		#dob_year_pattern = r'([0-9]{4})'
		#dob_year = re.match(dob_year_pattern, result)
		name_pattern = r'(\b[A-Z]{4,}\b|\b[A-Z]{1}[a-z]{2,}\b)'
		search_name_pattern = re.findall(name_pattern, str_name)
		search_name_pattern = search_name_pattern
		if not bool(search_name_pattern):
			print("namenotfound")
			name_pattern = r'(\b[A-Z]{4,}\b|\b[A-Z]{1}[a-z]{2,}\b)'
			search_name_pattern = re.findall(name_pattern, strr)

		#print(name_pattern[0:3]) # Returns found object
		aadhar_number_pattern = r'([0-9]{4}|[0-9]{12})'
		search_result_aadhar_number_pattern = re.findall(aadhar_number_pattern, strr)
		#search_result_aadhar_number_pattern = str(search_result_aadhar_number_pattern)
		#search_result_aadhar_number_pattern = search_result_aadhar_number_pattern.replace(str(dob_year),'')
		search_result_aadhar_number_pattern = search_result_aadhar_number_pattern[-3:]



		result_dict["name"] = str(search_name_pattern[:3])

		#result_dict["dob"] = str(search_result_dob)
		result_dict["id_number"] = str(search_result_aadhar_number_pattern )
		





		#added by me later
		

		return result_dict


	if card_type == 4:
		print("pancardoldest")

		pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

		tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
		strr = preocr(threshed)
		str_name = preocr_name(card_type , threshed)

		
		'''
		from pytesseract import Output
		str1 = "start  "

		d = pytesseract.image_to_data(threshed, output_type=Output.DICT)
		print(d)
		n_boxes = len(d['level'])
		for i in range(n_boxes):
			if(d['text'][i] != "" and float(d['conf'][i] )>= 50.0 and d['text'][i] != " "):
				(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
				print(d['text'][i])
				print(d['conf'][i])
				print("******************************")
				str1 = str1 +"\n"+ d['text'][i]
				cv2.rectangle(threshedfile, (x, y), (x + w, y + h), (0, 255, 0), 2)

		cv2.imwrite('result.png', threshed)
		str1 = str1.replace("|", "")
		str1 = str1.replace(".", "")
		str1 = str1.replace(",", " ")
		print(str1)
		'''




		result = pytesseract.image_to_string((threshed), config = r'-l eng --psm 6')
	
		for word in result.split("\n"):
			if "”—" in word:
				word = word.replace("”—", ":")
				#normalize NIK
			if "NIK" in word:
				nik_char = word.split()
			if "D" in word:
				word = word.replace("D", "0")
			if "?" in word:
				word = word.replace("?", "7") 

		female_pattern = r'(\bFEMALE\b|\bFemale\b|\bfemale\b|\b/FEMALE\b|\b/Female\b|\b/female\b)'
		search_result_female_pattern = re.findall(female_pattern , strr)
		result_dict["gender"] =search_result_female_pattern
		if bool(search_result_female_pattern):
			strr = strr.replace(str(search_result_female_pattern), " ")

		if not bool(search_result_female_pattern):
			male_pattern = r'(\bMALE\b|\bMale\b|\bmale\b|\b/MALE\b|\b/Male\b|\b/male\b)'
			search_result_male_pattern = re.findall(male_pattern , strr)
			result_dict["gender"] =search_result_male_pattern
			strr = strr.replace(str(search_result_male_pattern), " ")

		name_pattern = r'(\b[A-Z]{4,}\b|\b[A-Z]{1}[a-z]{2,}\b)'
		search_name_pattern = re.findall(name_pattern, str_name)
		search_name_pattern = search_name_pattern
		if not bool(search_name_pattern):
			print("namenotfound")
			name_pattern = r'(\b[A-Z]{4,}\b|\b[A-Z]{1}[a-z]{2,}\b)'
			search_name_pattern = re.findall(name_pattern, strr)


		result = pytesseract.image_to_string((threshed),config = r'-l eng --psm 6')

		#for word in result.split("\n"):
		#	if "”—" in word:
		#		word = word.replace("”—", ":")
  			#normalize NIK
		#	if "NIK" in word:
		#		nik_char = word.split()
		#	if "D" in word:
		#		word = word.replace("D", "0")
		#	if "?" in word:
		#		word = word.replace("?", "7") 
		#print(result.split())
		#print(result[1])
		date_pattern = r'([0-9]{2}\/[0-9]{2}\/[0-9]{4}|[0-9]{2}-[0-9]{2}-[0-9]{4})'
		search_result_dob = re.findall(date_pattern, strr)

		pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		search_result_pan_number_pattern = re.findall(pan_number_pattern, strr)
		#print(search_result_pan_number_pattern) # Returns pan number
		#exit()
		result_dict["name"] = search_name_pattern[:3]
		result_dict["dob"] = search_result_dob
		result_dict["id_number"] = search_result_pan_number_pattern 

		return result_dict



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
	# initialize the dimensions of the image to be resized and
	# grab the image size
	dim = None
	(h, w) = image.shape[:2]

	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (int(w * r), height)

	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))

	# resize the image
	resized = cv2.resize(image, dim, interpolation = inter)

	# return the resized image
	return resized


def object_detector(images_path):
	#im1 = Image.open(images_path) today
	im1 = images_path
	#im1 = Image.open('a.png') 
	# save a image using extension
	im1 = im1.save("D:/fortiate/project/static/uploads/geeks.png",dpi=(300.0, 300.0))

	img_org = cv2.imread('D:/fortiate/project/static/uploads/geeks.png',0)
	img_org_color = cv2.imread('D:/fortiate/project/static/uploads/geeks.png')

	#img_bw = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
	#img_bw = cv2.cvtColor(img_bw, cv2.COLOR_RGB2GRAY)
	print("nnn")


	card = find_card_type(img_org_color)     
	print(card , " -- card_type")
	#img_bw = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
	threshed = intergral_thresh(img_org)
	edged = crop_by_edges(threshed)

	#threshedfile = cv2.imread('nowwwwwww2thresh.jpg')


	  
	# creating a image object (main image) 

	

	cv2.imwrite("nowwwwwww2thresh.jpg", threshed)
	cv2.imwrite("nowwwwwww2edged.jpg", edged)

	#img_resized = image_resize(threshed, width = 1600, inter = cv2.INTER_AREA)
	img_resized = image_resize(edged, width = 1600, inter = cv2.INTER_AREA)


	dictt = ocr(card, img_resized)

	print(dictt)

	return dictt






if __name__ =='__main__':
	images_path = 'a.png'
	result = object_detector(images_path)

	






















































'''

#imgg_path = "D:/fortiate/project/IMG_20210415_163829.jpg"
#imgg_path = "D:/fortiate/project/aaaaa.jpg"
#imgg_path = "D:/fortiate/project/riyasat.jpg"


#img_org = cv2.imread(imgg_path)

#img_plate = crop_by_edges(img_org)
#cv2.imshow("bbb", img_plate)
#cv2.waitKey(0)
cv2.imshow("bbp",thresh)






'''
































































































