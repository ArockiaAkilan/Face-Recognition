#Importing the libraries and packages
from cv2 import distanceTransform
from sqlalchemy import null
import streamlit as st
import numpy as np
import os
import face_recognition as fcrn
import cv2 as cv
from PIL import Image
button=False

#creating a lists for images and classes(names)
path='Images'
Images=os.listdir(path)
images=[]
classes=[]
for i in Images:
    img=cv.imread(f'{path}/{i}')
    cv.cvtColor(img,cv.COLOR_BGR2RGB)
    images.append(img)
    classes.append(os.path.splitext(i)[0])

#A function to store encodings of training images
def encodings(images):
    encoding=[]
    for i in images:
        encoding.append(fcrn.face_encodings(i)[0])
    return encoding
encodingList=encodings(images)

#API Title and file uploader
st.title('Face Recognition')
st.write('Upload an image to recognise the face')
image=st.file_uploader('',type=['jpg','jpeg','png'])

#Finding an encodings of an uploaded image
if image :
    image=Image.open(image)
    image=np.array(image.convert('RGB'))
    locations=fcrn.face_locations(image)
    encoding=fcrn.face_encodings(image,locations)
    button=st.button('Recognize a face','button')

#Face recognition using  face_distance function
if button:
    for testLoc,testEncoding in zip(locations,encoding):
        matches=fcrn.compare_faces(encodingList,testEncoding)
        distance=fcrn.face_distance(encodingList,testEncoding)
    index=np.argmin(distance)
    if matches[index]:
        name=classes[index]
    image = cv.rectangle(image,(testLoc[3],testLoc[0]),(testLoc[1],testLoc[2]),(0,255,0),5)
    if matches[index]:
        st.title(name)
    else:
        st.title('Unknown')
    st.image(image)