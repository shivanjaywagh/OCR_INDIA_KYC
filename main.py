import cv2
import sys
import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import argparse 
import pytesseract 
from PIL import ImageGrab
from newnow3copy import object_detector
gender = 0
def filename_split(filename):
	return os.path.splitext(filename)[0]
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
def face(filename):
	#imagePath =url_for('static', filename=r'uploads/' + r'filename' + r'.' + r'jpg')
	#imagePath =url_for('static', filename='uploads/' + filename)
	#file_path = r'/static/uploads' + filename
	image = cv2.imread(filename)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	faceCascade = cv2.CascadeClassifier( os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml") )
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.3,
		minNeighbors=3,
		minSize=(30, 30)
	)
	for (x, y, w, h) in faces:
		###cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
		roi_color = image[y:y + h , x:x + w]
		#cv2.imwrite('static' +filename_split(filename) + '_face'+ os.path.splitext(filename)[1], roi_color)
		cv2.imwrite('face.jpg', roi_color)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')
@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#filename = filename_split(filename)
		face(os.path.join(app.config['UPLOAD_FOLDER'], filename))

		ret_dict = object_detector(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		######
		######
		#print('upload_image filename: ' + filename)
		return render_template('data.html', filename=filename,name=ret_dict["name"],dob=ret_dict["dob"],aadhaarno=ret_dict["id_number"],gender=ret_dict["gender"])
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename): 
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
@app.route('/display_face/<filename>')
def display_face(filename):
	filename_face=os.path.splitext(filename)[0] + '_face' + os.path.splitext(filename)[1]
	return redirect(url_for('static', filename=filename_face))
if __name__ == "__main__":
	app.debug = True
	app.run()
