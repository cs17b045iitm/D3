import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("video.avi")
template = cv2.imread('template.PNG',0)
w, h = template.shape[::-1]
ret, frame = cap.read()
cv2.imshow('image',frame)
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
w_r, h_r = img_gray.shape[::-1]

threshold = 0.7
i = 0
while True:
	ret, frame = cap.read()
	if (i%100)==0:
		img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		w1 = w
		h1 = h
		while h1<=h_r:
			template1 = cv2.resize(template, (int(w1), int(h1)), interpolation=cv2.INTER_CUBIC)
			res = cv2.matchTemplate(img_gray,template1,cv2.TM_CCOEFF_NORMED)
			loc = np.where( res >= threshold)
			for pt in zip(*loc[::-1]):
				cv2.rectangle(frame, pt, (pt[0] + w1, pt[1] + h1), (0,0,255), 2)

			w1 += int((1.0*w)/h)
			h1 += 1
			cv2.imshow('res.png',frame)
	#else
	i+=1
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.waitKey(0)
