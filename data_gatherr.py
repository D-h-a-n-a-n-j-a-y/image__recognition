#import cv2 as c
import os as o
import numpy as np
import cv2 as c

facecascade = c.CascadeClassifier(r'C:\Users\Dhananjay\Desktop\python\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
#id = input("Enter the id : ")
id='dj'
cap = c.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
count=0
while(True):
    ret, img = cap.read()
    gray = c.cvtColor(img, c.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces :
        c.rectangle(img,(x, y), (x+w, y+h), (255,255,255), 2)
        count+=1
        try:
            c.imwrite("Datasets/User_"+str(count)+"_"+str(id)+".jpg", gray[y:y+h, x:x+w])
        except IOError:
            pass
        c.imshow(str(id),img)
    k = c.waitKey(30) & 0xff
    if k == 27:
        break
    elif count>=100:
        break
cap.release()
c.destroyAllWindows()