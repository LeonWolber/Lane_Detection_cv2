import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

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


def canny_(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 50, 150)
	return canny

def roi(image):
	height = image.shape[0]
	width = image.shape[1]
	triangle = np.array([[(130, height), (272,133),(303,127), (width,260)]])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, triangle, 255)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def display_lines(image, lines):
	line_image = np.zeros_like(image)

	if lines is not None:
		for x1, y1, x2, y2  in lines:
			cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)



	return line_image


def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1*(2/5))
	x1 = int((y1 - intercept) / slope)
	x2 = int((y2 - intercept) / slope)

	return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
	left_fit = []
	right_fit = []

	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1,x2), (y1,y2), 1) # linear fct.
		slope = parameters[0]
		intercept = parameters[1]

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




if __name__ == '__main__':

	#while True:
		#image = grab_screen(region=(0,40,1920,1080))
		#cv2.imshow('window', cv2.resize(screen,(640,360)))
	image =  cv2.imread('vutwuT1.jpg')
	lane_image = np.copy(image)


	canny = canny_(lane_image)
	cropped_image = roi(canny)

	lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=50)
	averaged_lines = average_slope_intercept(lane_image, lines)
	line_image=  display_lines(lane_image, averaged_lines)
	combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1,1 )
	cv2.imshow('result', combo_image)


	if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			cv2.waitKey(0)
			break
