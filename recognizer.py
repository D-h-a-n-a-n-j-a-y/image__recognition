import cv2 as c
import numpy as np
import os as o

recognizer = c.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = c.CascadeClassifier(r'C:\Users\Dhananjay\Desktop\python\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')
font = c.FONT_HERSHEY_PLAIN

id = 0
names = ['NONE', 'DJ']

vid = c.VideoCapture(0)
vid.set(3,640)
vid.set(4,480)

mwwin = 0.1*vid.get(3)
mhwin = 0.1*vid.get(4)
while (True):
    ret, img = vid.read()
    img = c.flip(img, 1)
    gray = c.cvtColor(img, c.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.2,5,minSize=(int(mwwin), int(mhwin)))
    for(x, y, w, h) in faces:
        c.rectangle(img, (x, y), (x+w,y+h), (0, 0,233), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence < 100):
           id = names[id]
           confidence = "  {0}%".format(round(100 - confidence))
        else:
           id = "unknown"
           confidence = "  {0}%".format(round(100 - confidence))

        c.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        c.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    c.imshow('camera', img)
    k = c.waitKey(0) & 0xff
    if k == 27:
       break
    vid.release()
    c.destroyAllWindows()