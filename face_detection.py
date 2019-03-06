import numpy as np
import cv2 as c
faceCascade = c.CascadeClassifier(r'C:\Users\Dhananjay\Desktop\python\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
eyecascade = c.CascadeClassifier(r'C:\Users\Dhananjay\Desktop\python\Lib\site-packages\cv2\data\haarcascade_eye.xml')
cap=c.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while(True):
    ret, img = cap.read()
    img = c.flip(img,2)
    gray = c.cvtColor(img,c.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    #faces = faceCascade.load(filename = 'haarcascade_frontalface_default.xml')
    for (x,y,w,h) in faces:
        img = c.rectangle(img, (x,y), ((x+w), y+h), (255, 0, 0), 2)
       # img = c.ellipse(img, (x, y), 0, (255,255,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eyecascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes :
           # c.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,0,255), 1)
            c.circle(roi_color,((ex+int(ew/2)), ey+(int(eh/2))), 10, (0,255,0), 1)
    c.imshow('video',img)
    k = c.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
c.destroyAllWindows()