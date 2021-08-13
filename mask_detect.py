# 0:'MASK',1:'NO MASK'
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

face_model = cv2.CascadeClassifier(r'C:\Users\Pratham Sahay\Documents\opencv\haarcascade_frontalface_default.xml')
model = load_model(r'C:\Users\Pratham Sahay\Documents\Face Mask Detection\model\maskdet.h5')
img = cv2.imread(r'C:\Users\Pratham Sahay\Documents\Face Mask Detection\fm1.jpg')
gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
faces = face_model.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=4)
for i in range(len(faces)):
    (x,y,w,h) = faces[i]
    face_crop = img[y:y+h,x:x+w]
    face_crop = cv2.resize(face_crop,(128,128))
    face_crop = np.reshape(face_crop,[1,128,128,3])/255.0
    mask_result = model.predict(face_crop)
    if mask_result.argmax()==1:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.putText(img,'NO MASK',(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
    elif mask_result.argmax()==0:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(img,'MASK',(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
          
final_img = cv2.resize(img,(800,400))
cv2.imshow('Face Detection',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
