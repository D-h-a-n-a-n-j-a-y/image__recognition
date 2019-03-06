import numpy as np
import cv2 as c

cap=c.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while(True):
    ret, frame = cap.read()
    frame = c.flip(frame,1)
    gray = c.cvtColor(frame,c.COLOR_BGR2GRAY)

    c.imshow('frame',frame)
    c.imshow('gray',gray)

    k = c.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
c.destroyAllWindows()