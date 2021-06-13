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

import warnings
warnings.filterwarnings("ignore")



def preocr(threshedfile):
	str1 = "start  "

	from pytesseract import Output
	d = pytesseract.image_to_data(threshedfile, output_type=Output.DICT)
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

	cv2.imwrite('result.png', threshedfile)
	str1 = str1.replace("|", "")
	str1 = str1.replace(".", "")
	str1 = str1.replace(",", " ")
	str1 = str1.replace("asta", " ")
	str1 = str1.replace("fara", " ")
	str1 = str1.replace("are", " ")
	str1 = str1.replace("Father", " ")
	str1 = str1.replace("Name", " ")
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


	str1 = str1.replace(" ", "")













	print(str1)
	return str1














def crop_by_edges(img):
	#img_org = cv2.imread(images_path)
	
	img_org = img
	
	#print(type(img))
	size = np.shape(img_org)
	img_org2 = img_org.copy()
	img_bw = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

	ret3,img_thr = cv2.threshold(img_bw,90,255,cv2.THRESH_TOZERO)
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
		if (len(approx) == 4 and (aspect_ratio<1.8 and aspect_ratio>1.2)):
			screenCnt = approx
			print("screenCNt below:")
			print(screenCnt)
			break
	if screenCnt == []:
		print("no contours")
		# deskew
		return img_org
	#drawing contour around card plate
	mask = np.zeros(img_bw.shape, dtype=np.uint8)
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
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			#print(class_id)    #added by me
			confidence = scores[class_id]
			if confidence > 0.3:
				#pan2 = 1       # 2 --> pan2
				card_type = 2



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
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			#print(class_id)    #added by me
			confidence = scores[class_id]
			if confidence > 0.4:
				# Object detected
				#pan1 = 1       # 1 --> pan1 
				card_type = 1



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
		
	
		
		
		#th, threshed = cv2.threshold(cropped, 127, 255, cv2.THRESH_TRUNC)
		#hImg, wImg,_ = img.shape
		## (3) Detect
		result = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		strr = preocr(threshed)

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
				

		name_pattern = r'([A-Z]{4,})'
		search_name_pattern = re.findall(name_pattern, strr)

		result_dict["name"] = search_name_pattern
		
		date_pattern = r'([0-9]{2}\/[0-9]{2}\/[0-9]{4})'
		search_result_dob = re.findall(date_pattern, strr)    

		pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		search_result_pan_number_pattern = re.findall(pan_number_pattern, strr)

		result_dict["dob"] = search_result_dob
		result_dict["id_number"] = search_result_pan_number_pattern

		return result_dict

	if card_type == 2:
		pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
		tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
		

		result = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
		strr = preocr(threshed)
		
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

		name_pattern = r'([A-Z]{4,}|[A-Z][a-z]{3,})'
		search_name_pattern = re.findall(name_pattern, strr)
		search_name_pattern = search_name_pattern


		result = pytesseract.image_to_string((threshed),config = r'-l eng+hin --psm 6')

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

		pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		search_result_pan_number_pattern = re.findall(pan_number_pattern, strr)
		#print(search_result_pan_number_pattern) # Returns pan number
		#exit()
		result_dict["name"] = search_name_pattern
		result_dict["dob"] = search_result_dob
		result_dict["id_number"] = search_result_pan_number_pattern 

		return result_dict


	if card_type== 3:
		pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
		tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
		
		
		## (3) Detect
		result = pytesseract.image_to_string((threshed),config = r'-l eng+hin --psm 6')
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
		date_pattern = r'([0-9]{2}\/[0-9]{2}\/[0-9]{4})'
		search_result_dob = re.findall(date_pattern, strr)
		#print(search_result_dob) # Returns found object
		result_dict["dob"] = str(search_result_dob)
		if len(search_result_dob) == 0:
			dob_year_pattern = r'([0-9]{4})'
			dob_year = re.match(dob_year_pattern, strr)
			#print(dob_year_pattern)
			result_dict["dob"] = dob_year

		#dob_year_pattern = r'([0-9]{4})'
		#dob_year = re.match(dob_year_pattern, result)
		name_pattern = r'([A-Z]+[a-z]{3,})'
		name_pattern = re.findall(name_pattern, strr)
		#print(name_pattern[0:3]) # Returns found object
		aadhar_number_pattern = r'([0-9]{4}|[0-9]{12})'
		search_result_aadhar_number_pattern = re.findall(aadhar_number_pattern, strr)
		#search_result_aadhar_number_pattern = str(search_result_aadhar_number_pattern)
		#search_result_aadhar_number_pattern = search_result_aadhar_number_pattern.replace(str(dob_year),'')
		#search_result_aadhar_number_pattern = search_result_aadhar_number_pattern[1:]



		result_dict["name"] = str(name_pattern)
		#result_dict["dob"] = str(search_result_dob)
		result_dict["id_number"] = str(search_result_aadhar_number_pattern )

		#added by me later
		

		return result_dict


	if card_type == 4:
		pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

		tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
		

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




		result = pytesseract.image_to_string((threshed), config = r'-l eng+hin --psm 6')
	
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

		name_pattern = r'([A-Z]{4,}|[A-Z][a-z]{3,})'
		search_name_pattern = re.findall(name_pattern, str1)
		search_name_pattern = search_name_pattern


		result = pytesseract.image_to_string((threshed),config = r'-l eng+hin --psm 6')

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
		search_result_dob = re.findall(date_pattern, str1)

		pan_number_pattern = r'([A-Z]{5}[0-9]{4}[A-Z]{1})'
		search_result_pan_number_pattern = re.findall(pan_number_pattern, str1)
		#print(search_result_pan_number_pattern) # Returns pan number
		#exit()
		result_dict["name"] = search_name_pattern
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









if __name__ =='__main__':
	img_org = cv2.imread('IMG_20210415_163829.jpg')
	#img_bw = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
	#img_bw = cv2.cvtColor(img_bw, cv2.COLOR_RGB2GRAY)
	print("nnn")


	card = find_card_type(img_org)
	img_bw = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
	threshed = intergral_thresh(img_bw)

	#threshedfile = cv2.imread('nowwwwwww2thresh.jpg')
	edged = crop_by_edges(img_org)

	cv2.imwrite("nowwwwwww2thresh.jpg", threshed)
	cv2.imwrite("nowwwwwww2edged.jpg", edged)

	img_resized = image_resize(threshed, width = 1600, inter = cv2.INTER_AREA)

	dictt = ocr(card, img_resized)
	#img_org2 = cv2.imread(threshed)



	#imggg = Image.fromarray(threshed, 'L')



	
	#print(card)
	#result = pytesseract.image_to_string((threshed))
	#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR//tesseract.exe'

	#tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR//tessdata"'
	#result = pytesseract.image_to_string(threshed, lang ='eng', config=tessdata_dir_config)
	#result2 = pytesseract.image_to_boxes(threshed, lang ='eng', config=tessdata_dir_config)


	#print("1")
	#print(result)
	#print("2")

	#dictt = ocr(card, threshed)

	
	
	#text=re.findall(r'\\d+',result)     # remove one \
	#print(text[-3:])

	#print("resultdict:")
	#print(dictt)

	#print(result2)

	


	

	
	#for x in text:
	#	if(len(x)==8):
	#		datee={'day':x[:2],'month':x[2:4],'year':x[4:]}
	#	elif(len(x)==12):
	#		aadharno=x
	
	#print(datee)
	#print("date")
	#print(aadharno)
	#print("aadharno")

	print(dictt)























































'''

#imgg_path = "D:/fortiate/project/IMG_20210415_163829.jpg"
#imgg_path = "D:/fortiate/project/aaaaa.jpg"
#imgg_path = "D:/fortiate/project/riyasat.jpg"


#img_org = cv2.imread(imgg_path)

#img_plate = crop_by_edges(img_org)
#cv2.imshow("bbb", img_plate)
#cv2.waitKey(0)
'''
































































































