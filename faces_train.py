import os
import cv2 as cv 
import numpy as np 

people=["ben_afflek", "elton_john", "jerry_seinfeld", "madonna","mindy_kaling"]

DIR = r'C:\Users\marshal\desktop\opencv\train'

features =[]
labels=[]

haar_cascade = cv.CascadeClassifier('haar_face.xml')


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array =cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

            face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        #we have detected the faces now cropping out them
            for (x,y,w,h) in face_rect:
                #fae_roi region of interest
                face_roi =gray[y:y+h, x:x+w]
                features.append(face_roi)
                labels.append(label)#appending labels for computer to reduce strain while searchinfg for feature or blabla idk
create_train()
#converting features and list to np
features = np.array(features, dtype="object")

labels=np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

#saving this trained model so we are going to use it when evr we want it whereevre

#instantizizing the recognizer


#traIN THE recognizer on features and labels


# print(f'length of features ={len(features)}')
# print(f"length of labels ={len(labels)}")


