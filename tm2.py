import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('image_bg.PNG')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('template.PNG',0)
w, h = template.shape[::-1]
w_r, h_r = img_gray.shape[::-1]
w1 = w
h1 = h
threshold = 0.7
while h1<=h_r:
	template1 = cv2.resize(template, (int(w1), int(h1)), interpolation=cv2.INTER_CUBIC)
	res = cv2.matchTemplate(img_gray,template1,cv2.TM_CCOEFF_NORMED)
	loc = np.where( res >= threshold)
	for pt in zip(*loc[::-1]):
		cv2.rectangle(img_rgb, pt, (pt[0] + w1, pt[1] + h1), (0,0,255), 2)

	w1 += int((1.0*w)/h)
	h1 += 1
cv2.imshow('res.png',img_rgb)
cv2.waitKey(0)