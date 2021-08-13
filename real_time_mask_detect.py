from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import argparse

face_model = cv2.CascadeClassifier(r'C:\Users\Pratham Sahay\Documents\opencv\haarcascade_frontalface_default.xml')
model = load_model(r'C:\Users\Pratham Sahay\Documents\Face Mask Detection\model\maskdet.h5')
cap=cv2.VideoCapture(0)
while cap.isOpened():
    _,frame=cap.read()
    faces = face_model.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=4)
    for (x,y,w,h) in faces:
        crop_face = frame[y:y+h,x:x+w]
        crop_face = cv2.resize(crop_face,(128,128))
        crop_face = np.reshape(crop_face,[1,128,128,3])/255.0
        mask_result = model.predict(crop_face)
        if mask_result.argmax()==1:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(frame,'NO MASK',(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        elif mask_result.argmax()==0:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            cv2.putText(frame,'MASK',(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.imshow('Mask Detection',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()