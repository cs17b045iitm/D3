import numpy as np
import cv2 as cv
cap = cv.VideoCapture('license.mp4')
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
drawing = False # true if mouse is pressed
w, h = 0,0
r,c = -1,-1

print "Draw (invisible) rectangle around region of interest and hit space to start program" 
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global r,c,h,w,drawing

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        r,c = y,x

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
        	h = -(r + h) + y
        	w = -(c + w) + x 
			#cv2.rectangle
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        

#img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)

while(1):
    cv.imshow('image',frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord(' '):
    	#print r, c, w, h
        break

track_window = (c,r,w,h)
#print track_window
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
	cv.imshow('hist',dst)
	cv.waitKey(0)
        # apply meanshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        cv.imshow('img2',img2)
        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k)+".jpg",img2)
    else:
        break
cv.destroyAllWindows()
cap.release()
