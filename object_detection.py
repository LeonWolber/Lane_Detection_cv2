import cv2
import matplotlib.pyplot as plt



config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)





classLabels = []

file_name = 'labels.txt'

with open(file_name, 'rt') as fpt:
	classLabels = fpt.read().rstrip('\n').split('\n')
	classLabels.append(fpt.read())



img = cv2.imread('Unbenannt.JPG')
# cv2.imshow('',img)
# cv2.waitKey(0)




ClassIndex, confidence, bbox = model.detect(img, confThreshold = 0.5)
print(ClassIndex)


font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN	
for ind, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
	cv2.rectangle(img, boxes, (0,255,0), 1)
	cv2.putText(img, classLabels[ind-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color = (0,255,0), thickness = 1)


cv2.imshow('',img)
cv2.waitKey(0)