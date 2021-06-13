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
exit = 0
def remove_noise_and_smooth(file_name):
	#img = set_image_dpi(file_name)
	img = cv2.imread(file_name, 0)
	filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	blur = cv2.GaussianBlur(filtered,(5,5),0)
	ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	median = cv2.medianBlur(th3,5)
	median = cv2.medianBlur(median,5)
	median = cv2.medianBlur(median,5)
	median = cv2.medianBlur(median,5)
	median = cv2.bitwise_not(median)
	kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
	img_dil = cv2.dilate(median, kernel, iterations = 1)
	median = cv2.bitwise_not(img_dil)
	return median
# Images path
#images_path = glob.glob(r"D:/fortiate/Google-Image-Scraper-master-Copy/new_pan/*.jpg")
#simgg_path = "D:/fortiate/IMG_20210415_163829.jpg"
imgg_path = "D:/fortiate/a-copy.jpg"

flag = 0
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) 
kernel = 1/5 * kernel
result_list = []
result_dict = {}


def object_detector(images_path):
	# Insert here the path of your images
	#random.shuffle(images_path)
	# loop through all the images
	#img_org = cv2.imread(images_path)
	img_org = remove_noise_and_smooth(images_path)
	size = np.shape(img_org)
	img_org2 = img_org.copy()
	#img_bw = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
	#sharpened = cv2.filter2D(img_bw, -1, kernel)
	#cv2.imwrite('sharpened.jpg', sharpened)
	#img_blur = cv2.GaussianBlur(img_bw,(5,3), 1)     #added by me
	#img_smooth = remove_noise_and_smooth(img_path)
	#ret3,img_thr = cv2.threshold(img_bw,90,255,cv2.THRESH_TOZERO)
	#th3 = cv2.adaptiveThreshold(img_thr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	#cv2.imwrite('thresh.jpg',img_thr)
	#Canny's Edge Detection
	img_edg  = cv2.Canny(img_org,10,50)
	#cv2.imwrite('cn_edge.jpg' , img_edg)
	#increasing the thickness of the edges
	kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (7, 7))
	img_dil = cv2.dilate(img_edg, kernel, iterations = 1)
	#cv2.imwrite('dilated_img.jpg',img_dil)
	#finding the contours and sorting them in descending order
	(contours ,hierarchye) = cv2.findContours(img_dil.copy(), 1, 2)
	cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
	#giving shape to the contours
	screenCnt = None 
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
		if (len(approx) == 4 and (aspect_ratio<1.8 and aspect_ratio>1.2)):
			screenCnt = approx
			#print(screenCnt)
			break
	#drawing contour around card plate
	mask = np.zeros(img_org2.shape, dtype=np.uint8)
	roi_corners = np.array(screenCnt ,dtype=np.int32)
	ignore_mask_color = (255,)*1
	cv2.fillPoly(mask, roi_corners , ignore_mask_color)
	cv2.drawContours(img_org, [screenCnt], -40, (100, 255, 100), 9)
	#cv2.imshow('original  image with boundry' , img_org)
	#cv2.imwrite('plate_detedted.jpg',img_org)
	ys =[screenCnt[0,0,1] , screenCnt[1,0,1] ,screenCnt[2,0,1] ,screenCnt[3,0,1]]
	xs =[screenCnt[0,0,0] , screenCnt[1,0,0] ,screenCnt[2,0,0] ,screenCnt[3,0,0]]
	ys_sorted_index = np.argsort(ys)
	xs_sorted_index = np.argsort(xs)
	x1 = screenCnt[xs_sorted_index[0],0,0]
	x2 = screenCnt[xs_sorted_index[3],0,0]
	y1 = screenCnt[ys_sorted_index[0],0,1]
	y2 = screenCnt[ys_sorted_index[3],0,1]
	img_plate = img_org2[y1:y2 , x1:x2]
	#img_plate = cv2.fastNlMeansDenoisingColored(img_plate,None,10,10,7,21)  #added by me
	#cv2.imshow('number_plate',img_plate)
	# Load Yolo pan2
	net = cv2.dnn.readNet("yolov3_training_last_pan2new.weights", "yolov3_testing.cfg")
	# Name custom object
	classes = ["pancard_2"]
	pan2=0
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	'''
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	'''
	#img = cv2.imread(img_plate)
	img = cv2.resize(img_plate, None, fx=0.4, fy=0.4)
	height, width = img.shape
	# Detecting objects
	blob = cv2.dnn.blobFromImage(img_plate, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)
	#print(outs)#added by me
	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			#print(class_id)    #added by me
			confidence = scores[class_id]
			if confidence > 0.3:
				# Object detected
				pan2 = 1       # 2 --> pan2
				#print("pancard_2")
				#print(confidence)    #added by me
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
				indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
				#print(indexes)
				font = cv2.FONT_HERSHEY_PLAIN
				for i in range(len(boxes)):
					if i in indexes:
						x, y, w, h = boxes[i]
						label = str(classes[class_ids[i]])
						color = colors[class_ids[i]]
						#cv2.rectangle(img_plate, (x, y), (x + w, y + h), color, 2)
						#cv2.putText(img_plate, label, (x, y + 30), font, 1, color, 2)
				#cv2.imshow("Image", img_plate)
				#key = cv2.waitKey(0)
				#cv2.destroyAllWindows()
		# ocr pan2
	# if pancard 2 , then this code block for ocr
	if pan2 == 1:
		#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
		#tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
		#pytesseract.image_to_string(image, config=tessdata_dir_config)
		## (1) Read
		#img = cv2.imread(img_plate)
		gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
		
		height, width, channels = img_plate.shape
		xxx1 = width*0.0382
		xxx2 = width*0.347
		yyy1 = height*0.5525
		yyy2 = height*0.6137
		#cropped = gray[103:163,36:186]
		cropped = gray[int(yyy1):int(yyy2),int(xxx1):int(xxx2)]
		cv2.imwrite('img.jpg', cropped)
		#cv2.imshow('img',cropped)
		#cv2.waitKey(0)
		'''
		th, threshed = cv2.threshold(cropped, 127, 255, cv2.THRESH_TRUNC)
		hImg, wImg,_ = img.shape
		## (3) Detect
		result_name = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		detect_langs(result_name)
		for word in result_name.split("\n"):
			if "”—" in word:
				word = word.replace("”—", ":")
  				
				#normalize NIK
			if "NIK" in word:
				nik_char = word.split()
			if "D" in word:
				word = word.replace("D", "0")
			if "?" in word:
				word = word.replace("?", "7") 
				
		print(result_name.split())
		'''
		#ocr part
		## (2) Threshold
		#cropped = gray[int(y):,:]
		#cv2.imshow('img',cropped)
		#cv2.waitKey(0)
		th, threshed = cv2.threshold(cropped, 127, 255, cv2.THRESH_TRUNC)
		hImg, wImg,_ = img.shape
		## (3) Detect
		result = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		#detect_langs(result)
		#print(result)
		## (5) Normalize
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
		#date_pattern = r'([0-9]{2}\/[0-9]{2}\/[0-9]{4})'
		#search_result_dob = re.findall(date_pattern, result)    
		#print(search_result_dob) # Returns dob
		name_pattern = r'([A-Z]{4,})'
		search_name_pattern = re.findall(name_pattern, result)
		#search_name_pattern = search_name_pattern[3:]
		#print(name_pattern[0:3]) # Returns found object
		#pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		#search_result_pan_number_pattern = re.findall(pan_number_pattern, result)
		#print(search_result_pan_number_pattern) # Returns pan number
		#exit()
		#result_list.append(name_pattern)
		#result_list.append(date_pattern)
		#result_list.append(name_pattern)
		result_dict["name"] = search_name_pattern
		#result_dict["dob"] = search_result_dob
		#result_dict["id_number"] = search_result_pan_number_pattern



		th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
		hImg, wImg,_ = img.shape
		## (3) Detect
		result = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		#detect_langs(result)
		#print(result)
		## (5) Normalize
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
		search_result_dob = re.findall(date_pattern, result)    
		#print(search_result_dob) # Returns dob
		#name_pattern = r'([A-Z]{4,})'
		#search_name_pattern = re.findall(name_pattern, result)
		#search_name_pattern = search_name_pattern[3:]
		#print(name_pattern[0:3]) # Returns found object
		pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		search_result_pan_number_pattern = re.findall(pan_number_pattern, result)
		#print(search_result_pan_number_pattern) # Returns pan number
		#exit()
		#result_list.append(name_pattern)
		#result_list.append(date_pattern)
		#result_list.append(name_pattern)
		#result_dict["name"] = search_name_pattern
		result_dict["dob"] = search_result_dob
		result_dict["id_number"] = search_result_pan_number_pattern








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
	img = cv2.resize(img_plate, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	# Detecting objects
	blob = cv2.dnn.blobFromImage(img_plate, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)
	#print(outs)#added by me
	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			#print(class_id)    #added by me
			confidence = scores[class_id]
			if confidence > 0.4:
				# Object detected
				pan1 = 1       # 1 --> pan1 
				#print("pancard_1")
				#print(confidence)    #added by me
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
				indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
				#print(indexes)
				font = cv2.FONT_HERSHEY_PLAIN
				for i in range(len(boxes)):
					if i in indexes:
						x, y, w, h = boxes[i]
						label = str(classes[class_ids[i]])
						color = colors[class_ids[i]]
						cv2.rectangle(img_plate, (x, y), (x + w, y + h), color, 2)
						cv2.putText(img_plate, label, (x, y + 30), font, 1, color, 2)
				#cv2.imshow("Image", img_plate)
				#key = cv2.waitKey(0)
				#cv2.destroyAllWindows()
	# ocr pan1
	# if pancard 1 , then this code block for ocr
	if pan1 == 1:
		pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
		tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
		#pytesseract.image_to_string(image, config=tessdata_dir_config)
		## (1) Read
		#img = cv2.imread(img_plate)
		gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
		height, width, channels = img_plate.shape
		xxx1 = width*0.0382				#percentage values based on template
		xxx2 = width*0.5640
		yyy1 = height*0.5075
		yyy2 = height*0.6137
		#cropped = gray[103:163,36:186]
		#ropped = gray[int(yyy1):int(yyy2),int(xxx1):int(xxx2)]
		#cropped = cropped[int(yyy1):int(yyy2),int(xxx1):int(xxx2)]
		#cv2.imshow('img',cropped)
		#cv2.waitKey(0)
		'''
		th, threshed = cv2.threshold(cropped, 127, 255, cv2.THRESH_TRUNC)
		hImg, wImg,_ = img.shape
		## (3) Detect
		result_name = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		detect_langs(result_name)
		for word in result_name.split("\n"):
			if "”—" in word:
				word = word.replace("”—", ":")
  				
				#normalize NIK
			if "NIK" in word:
				nik_char = word.split()
			if "D" in word:
				word = word.replace("D", "0")
			if "?" in word:
				word = word.replace("?", "7") 
				
		#print(result_name.split())
		'''
		#dob part
		## (2) Threshold
		#cropped = gray[int(y):,:]
		#cv2.imshow('img',cropped)
		#cv2.waitKey(0)


		th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
		hImg, wImg,_ = img.shape
		## (3) Detect
		result = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		#detect_langs(result)
		#print(result)
		## (5) Normalize
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
		3#date_pattern = r'([0-9]{2}\/[0-9]{2}\/[0-9]{4})'
		#search_result_dob = re.findall(date_pattern, result)
		#print(search_result_dob) # Returns found object
		name_pattern = r'([A-Z]{4,})'
		search_name_pattern = re.findall(name_pattern, result)
		search_name_pattern = search_name_pattern[5:]

		#rint(name_pattern[0:3]) # Returns found object
		#pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		#search_result_pan_number_pattern = re.findall(pan_number_pattern, result)
		#print(search_result_pan_number_pattern) # Returns pan number












		th, threshed = cv2.threshold(img_plate, 127, 255, cv2.THRESH_TRUNC)
		hImg, wImg,_ = img.shape
		## (3) Detect
		result = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		#detect_langs(result)
		#print(result)
		## (5) Normalize
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
		search_result_dob = re.findall(date_pattern, result)
		#print(search_result_dob) # Returns found object
		#name_pattern = r'([A-Z]+[a-z]{4,})'
		#name_pattern = re.findall(name_pattern, result)
		#rint(name_pattern[0:3]) # Returns found object
		pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		search_result_pan_number_pattern = re.findall(pan_number_pattern, result)
		#print(search_result_pan_number_pattern) # Returns pan number
		#exit()
		result_dict["name"] = search_name_pattern
		result_dict["dob"] = search_result_dob
		result_dict["id_number"] = search_result_pan_number_pattern 















		#-----------*-----------------------*----------------------*-------------------
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
	img = cv2.resize(img_plate, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	# Detecting objects
	blob = cv2.dnn.blobFromImage(img_plate, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	outs = net.forward(output_layers)
	#print(outs)#added by me
	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			#print(class_id)    #added by me
			confidence = scores[class_id]
			if confidence > 0.3:
				# Object detected
				aadhar_flag = 1       # 3 --> aadhar
				#print("aadhar")
				#print(confidence)    #added by me
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)
				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
				indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
				#print(indexes)
				font = cv2.FONT_HERSHEY_PLAIN
				for i in range(len(boxes)):
					if i in indexes:
						x, y, w, h = boxes[i]
						label = str(classes[class_ids[i]])
						color = colors[class_ids[i]]
						#cv2.rectangle(img_plate, (x, y), (x + w, y + h), color, 2)
						#cv2.putText(img_plate, label, (x, y + 30), font, 1, color, 2)
				#cv2.imshow("Image", img_plate)
				#key = cv2.waitKey(0)
				#cv2.destroyAllWindows()
	# ocr aadhar
	# if aadhar , then this code block for ocr
	if aadhar_flag == 1:
		pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
		tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
		#pytesseract.image_to_string(image, config=tessdata_dir_config)
		## (1) Read
		#img = cv2.imread(img_plate)
		gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)
		height, width, channels = img_plate.shape
		xxx1 = width*0.275			#percentage values based on template
		xxx2 = width*0.8710
		yyy1 = height*0.2990
		yyy2 = height*0.3970
		#cropped = gray[103:163,36:186]
		cropped = gray[int(yyy1):int(yyy2),int(xxx1):int(xxx2)]
		#cv2.imshow('img',cropped)
		#cv2.waitKey(0)
		'''
		th, threshed = cv2.threshold(cropped, 127, 255, cv2.THRESH_TRUNC)
		hImg, wImg,_ = img.shape
		## (3) Detect
		result_name = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		detect_langs(result_name)
		for word in result_name.split("\n"):
			if "”—" in word:
				word = word.replace("”—", ":")
  				
				#normalize NIK
			if "NIK" in word:
				nik_char = word.split()
			if "D" in word:
				word = word.replace("D", "0")
			if "?" in word:
				word = word.replace("?", "7") 
				
		#print(result_name.split())
		'''
		#dob part
		## (2) Threshold
		#cropped = gray[103:163,36:186]
		#cropped = gray[int(y):,:]
		#cv2.imshow('img',cropped)
		#cv2.waitKey(0)
		th, threshed = cv2.threshold(img_plate, 127, 255, cv2.THRESH_TRUNC)
		hImg, wImg,_ = img.shape
		## (3) Detect
		result = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		#detect_langs(result)
		#print(result)
		## (5) Normalize
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
		search_result_dob = re.findall(date_pattern, result)
		#print(search_result_dob) # Returns found object
		result_dict["dob"] = str(search_result_dob)
		if len(search_result_dob) == 0:
			dob_year_pattern = r'([0-9]{4})'
			dob_year = re.match(dob_year_pattern, result)
			#print(dob_year_pattern)
			result_dict["dob"] = dob_year

		#dob_year_pattern = r'([0-9]{4})'
		#dob_year = re.match(dob_year_pattern, result)
		name_pattern = r'([A-Z]+[a-z]{4,})'
		name_pattern = re.findall(name_pattern, result)
		#print(name_pattern[0:3]) # Returns found object
		aadhar_number_pattern = r'([0-9]{4})'
		search_result_aadhar_number_pattern = re.findall(aadhar_number_pattern, result)
		#search_result_aadhar_number_pattern = str(search_result_aadhar_number_pattern)
		#search_result_aadhar_number_pattern = search_result_aadhar_number_pattern.replace(str(dob_year),'')
		search_result_aadhar_number_pattern = search_result_aadhar_number_pattern[1:]


		#print(search_result_aadhar_number_pattern) # Returns aadhar number
		#print("brekk")
		#print(result)
		#exit()
		result_dict["name"] = str(name_pattern)
		#result_dict["dob"] = str(search_result_dob)
		result_dict["id_number"] = str(search_result_aadhar_number_pattern )

	return result_dict


#dicc = object_detector(imgg_path)
#print(dicc)

















































































































