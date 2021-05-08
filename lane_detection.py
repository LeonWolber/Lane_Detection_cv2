import cv2
import numpy as np
import argparse
import win32gui, win32ui, win32con, win32api

import matplotlib.pyplot as plt
import time

CONFIG_FILE = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
FROZEN_MODEL = 'frozen_inference_graph.pb'
FILE_NAME = 'labels.txt'
FONT_SIZE = 3
FONT = cv2.FONT_HERSHEY_PLAIN



def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def roi(image):
	#height = image.shape[0]
	#width = image.shape[1]		

	#vertices = np.array([[(80,477), (310,355),(450, 327), (600,355),(852,470), (540, 446)]])
	vertices = np.array([[(180,430),(390,310),(510,300),(700,430)]])

	mask = np.zeros_like(image)
	cv2.fillPoly(mask, vertices, 255)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def canny_(image):
	gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 50, 150)
	return canny

def display_lines(image, lines):
	line_image = np.zeros_like(image)

	if lines is not None:
		for x1, y1, x2, y2  in lines:
			cv2.line(line_image, (x1,y1), (x2,y2), (255,0,255), 10)

	return line_image


def average_slope_intercept(image, lines):
	
	left_fit = []
	right_fit = []

	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		#print(x1, y1, x2, y2)
		parameters = np.polyfit((x1,x2), (y1,y2), 1) # linear fct.
		slope = parameters[0]
		intercept = parameters[1]
		#print(slope, intercept)

		# lines on left negative slope
		# lines on right positive slope

		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope, intercept))
	

	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)

	return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = 490
	y2 = int(y1*(5.3/8))
	x1 = int((y1 - intercept) / slope)
	x2 = int((y2 - intercept) / slope)

	return np.array([x1, y1, x2, y2])

def display_multiple_lines(image, lines):
	line_image = np.zeros_like(image)

	if lines is not None:
		for line  in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(line_image, (x1,y1), (x2,y2), (0,0,255), 10)

	return line_image

def load_model():

	model = cv2.dnn_DetectionModel(FROZEN_MODEL, CONFIG_FILE)
	model.setInputSize(320,320)
	model.setInputScale(1.0/127.5)
	model.setInputMean((127.5,127.5,127.5))
	model.setInputSwapRB(True)

	return model

def get_classes():
	classLabels = []

	with open(FILE_NAME, 'rt') as fpt:
		classLabels = fpt.read().rstrip('\n').split('\n')
		classLabels.append(fpt.read())

	return classLabels

def detect_objects(labels, img):
	class_index, confidence, bbox = model.detect(img, confThreshold = 0.45)
	if class_index[0] in [3,4,6,8]:

		for ind, conf, boxes in zip(class_index.flatten(), confidence.flatten(), bbox):
			cv2.rectangle(img, boxes, (0,255,255), 4)
			# cv2.putText(img,
			# 			 labels[ind-1],
			# 			  (boxes[0],boxes[1]+40),
			# 			    FONT,
			# 			     fontScale=FONT_SIZE,
			# 			      color = (0,255,0),
			# 			       thickness = 2)

	return img

def roi_object_detection(image):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	vertices = np.array([[(0, 300),(0, 450),(920,450),(920,300)]])

	mask = np.zeros_like(img)
	cv2.fillPoly(mask, vertices, color=255)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img

if __name__ == '__main__':



	model = load_model()
	classes = get_classes()
	#https://www.youtube.com/watch?v=T0IHvrEAAUs

	while True:
		last = time.time()
		screen = grab_screen(region=(0,40,1920,1080))
		crop_img = screen[130:640, 40:920]

		
		
		lane_image = np.copy(crop_img)
		roi_objects = roi_object_detection(lane_image)
		# plt.imshow(roi_objects)
		# plt.show()



		canny = canny_(lane_image)
		region_img = roi(canny)

		lines = cv2.HoughLinesP(region_img,
											 rho=6,
											 theta = np.pi/180,
											 threshold = 100,
											 minLineLength=0.1,
											 maxLineGap=75)
		
		
		try:
			
			averaged_lines = average_slope_intercept(lane_image, lines)
			line_image =  display_lines(lane_image, averaged_lines)

			#line_image =  display_multiple_lines(lane_image, lines)
			combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,0)

			final_img = cv2.cvtColor(detect_objects(labels = classes, img = lane_image), cv2.COLOR_BGR2RGB)
			final = cv2.addWeighted(combo_image, 1, final_img, 0.3,0)
			#final = increase_brightness(final, value=-60)
			cv2.imshow("cropped",final)
			# cv2.imshow("cropped", cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
		except:
			try:
			
		
				
				cv2.imshow("cropped", cv2.cvtColor(combo_image, cv2.COLOR_BGR2RGB))
			except:
				cv2.imshow("cropped", canny)
		# 	pass

		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			cv2.waitKey(0)
			break


	
