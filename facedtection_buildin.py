#build in

#face recognision is performed using classifirers is an algorithm to detect face +ve and -ve ie present or not and neeed to be trainde thousands and thousands of images however opencv it for us
#two types har classifier and more advanced classifiers they binary and etc.. but theyare less noise prone
#there are plenty available you can acess any in github page of opencv

import cv2 as cv 

img = cv.imread('group_photo.jpeg')

def rescaleFrame(frame, scale=0.25):
    width=int(frame.shape[1]*scale) #shape[1]-width
    height=int(frame.shape[0]*scale) #[0]-height
    dimensions=(width,height)
    
    return cv.resize(frame, dimensions , interpolation=cv.INTER_AREA)  #resize is for alll

img = rescaleFrame(img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#reading the haarcascade file
haar_cascade = cv.CascadeClassifier('haar_face.xml')
#haar cascade are really sensitive to noise

#detect mutiscale instance of classifier class
#use variables mutiscale and neighbour and essentially the rectangular coordinates of thatfae as a list to faes_rect
# you can find no. of face by printing this variable  

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)
print("no. of faces found are")#keep sale low for beeter result and 1.1 is min
print(len(face_rect))

#if you see output of the faces_rect it gives list ogf tuples of cordinats of the faces found x,y,w,h
for (x,y,w,h) in face_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow("detected face", img)

cv.waitKey(0)