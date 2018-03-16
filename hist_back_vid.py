import cv2
import numpy as np
roi = cv2.imread('template.PNG')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

cap = cv2.VideoCapture('video.avi')

while (1):
	ret,frame = cap.read()
	if (ret == True):
		target = frame
		#cv2.imshow('res.jpg',target)
		#cv2.waitKey(0)
		hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)

		# calculating object histogram
		roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )

		# normalize histogram and apply backprojection
		cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
		dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)


		#cv2.imshow('dst', dst)
		#cv2.waitKey(0)


		# Now convolve with circular disc
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		cv2.filter2D(dst,-1,disc,dst)
		#cv2.imshow('dst', dst)
		#cv2.waitKey(0)
		# threshold
		ret,thresh = cv2.threshold(dst,10,255,0) 

		#dilate, erode and find morph_gradient
		kernel = np.ones((17,17),np.uint8)
		thresh = cv2.dilate(thresh, kernel, iterations = 1)

		#cv2.imshow('closed', thresh)
		#cv2.waitKey(0)

		kernel1 = np.ones((5,5),np.uint8)
		thresh = cv2.erode(thresh, kernel1, iterations = 1)

		#cv2.imshow('closed', thresh)
		#cv2.waitKey(0) 
		
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel1)
		
		#cv2.imshow('closed', thresh)
		#cv2.waitKey(0)

		final = cv2.merge((thresh,thresh,thresh))
		final[:,:,2] = 0

		res = cv2.bitwise_or(target,final)

		cv2.imshow('video',res)
		k = cv2.waitKey(60) & 0xff
		if k == 27:
			break
	else:
		break
cv2.destroyAllWindows()
cap.release()
