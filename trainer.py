import cv2 as c
import numpy as np
import PIL.Image as i
import os as o

path = r"Datasets"
recognizer = c.face.LBPHFaceRecognizer_create()
detector = c.CascadeClassifier("haarcascade_frontalface_default.xml")

"""def imglabel(path):
    impaths = [o.path.join(path,f) for f in o.listdir(path)]
    facesample = []
    ids  = []
    for impath in impaths:
        pil_img = i.open(path).convert("L")
        img_numpy = np.array(pil_img, 'uint-8')
        id = int(o.path.split(path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            facesample.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return facesample, ids
"""
faceCascade = c.CascadeClassifier(r'C:\Users\Dhananjay\Desktop\python\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml')

def imglabel(path):
    impaths = []
    for im in range(100):
        impaths.append(path+"/User_"+str(im+1)+"_dj.jpg")
    facesample = []
    ids = []
    file = open("data.txt","a")
    for impath in impaths:
        print(impath)
        #pil_img = i.open(impath).convert("L")
        pil_img = c.imread(impath,c.IMREAD_GRAYSCALE)
        """c.imshow('aa',pil_img)
        c.waitKey(0)
        c.destroyAllWindows()"""
        """img_numpy = np.asarray(pil_img)

        img_numpy = img_numpy.tolist()

        img_numpy = str(impath)+"\n"+str(img_numpy)+"\n"
        file.write(img_numpy)
        #print(img_numpy)"""
        img_numpy = np.array(pil_img, 'uint8')
        #i = int(((o.path.split(impath)[-1].split(".")[1])))
        i= 1
        print(i)
        #faces = detector.detectMultiScale(img_numpy)
        faces = faceCascade.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            facesample.append(img_numpy[y:y + h, x:x + w])
            ids.append(i)
    return facesample, ids

print("\n -/-*******-\- LOADING -/-*******-\-")
faces, ids = imglabel(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')

print("\n model trained successfully......aborting".format(len(np.unique(ids))))

