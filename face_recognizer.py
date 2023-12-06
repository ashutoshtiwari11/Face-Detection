import numpy as np 
import cv2 as cv 

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people=["ben_afflek", "elton_john", "jerry_seinfeld", "madonna","mindy_kaling"]
feature =np.load('feature.npy', allow_pickle=True)
labels=np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\marshal\Desktop\opencv\Face Detection\data\val\ben_afflek\httpabsolumentgratuitfreefrimagesbenaffleckjpg.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("image",gray)

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)


for (x,y,w,h) in face_rect:
    face_roi =gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(face_roi)
    print(f'label={people[label]} with a confidence of{confidence}')
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow("detected face", img)
cv.waitKey(0)
 