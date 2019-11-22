# -*- coding: utf-8 -*-
"""
Created on Sun oct 13 22:54:17 2019

@author: KV
"""

import cv2
import matplotlib.pyplot as plt
import csv
import os

face_cascade_eye_right = cv2.CascadeClassifier('haarcascades/haarcascade_righteye_2splits.xml')
face_cascade_eye_left = cv2.CascadeClassifier('haarcascades/haarcascade_lefteye_2splits.xml')
face_cascade_nose = cv2.CascadeClassifier('haarcascades/nose.xml')


def draw_glasses(img1,bound_eye):
    if(len(bound_eye)>=2):
        #img = cv2.resize(img,(600,600))
        try:
            x_o = int(bound_eye[0][0] - 25)
            y_o = int(bound_eye[0][1] - 25)
            x_o = abs(x_o)
            y_o = abs(y_o)
            x_r = 2*((bound_eye[0][0]+bound_eye[1][0]+bound_eye[1][2])/2) - 2*x_o - 10
            y_r = max(bound_eye[1][3],bound_eye[0][3])+75
            x_r = abs(x_r)
            y_r = abs(y_r)
            img2 = cv2.imread('Data/Train/glasses.png',-1)
            img2 = cv2.resize(img2,(int(x_r),int(y_r)))
            
            y1, y2 = y_o, y_o + img2.shape[0]
            x1, x2 = x_o, x_o + img2.shape[1]
        
            alpha_s = img2[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
        
            for c in range(0, 3):
                try:
                    img1[y1:y2, x1:x2, c] = (alpha_s * img2[:, :, c] + alpha_l * img1[y1:y2, x1:x2, c])
                except():
                    pass
        except():
            pass
    return img1


def draw_mustache(img1,bound_nose):
    if(len(bound_nose)>=1):
        #img = cv2.resize(img,(600,600))
        try:
            x_o = int(bound_nose[0][0] - (bound_nose[0][3]/bound_nose[0][2])*5)
            y_o = int(bound_nose[0][1]+(bound_nose[0][3])/2 + 8)
            x_o = abs(x_o)
            y_o = abs(y_o)
            x_r = 1.5*(bound_nose[0][2])
            y_r = max(bound_nose[0][2]/2,0)
            x_r = abs(x_r)
            y_r = abs(y_r)
            img2 = cv2.imread('Data/Train/mustache_edited.png',-1)
            img2 = cv2.resize(img2,(int(x_r),int(y_r)))
            
            y1, y2 = y_o, y_o + img2.shape[0]
            x1, x2 = x_o, x_o + img2.shape[1]
        
            alpha_s = img2[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
        
            for c in range(0, 3):
                try:
                    img1[y1:y2, x1:x2, c] = (alpha_s * img2[:, :, c] + alpha_l * img1[y1:y2, x1:x2, c])
                except():
                    pass
        except():
            pass
    return img1


def image_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose,img1):
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    eye_r = face_cascade_eye_right.detectMultiScale(gray, 1.3, 5)    
    eye_l = face_cascade_eye_left.detectMultiScale(gray, 1.3, 5)
    nose = face_cascade_nose.detectMultiScale(gray, 1.3, 5)
    bound_eye = []
    bound_nose = []
    for (x,y,w,h) in eye_r:
        #cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
        bound_eye.append([x,y,w,h])
    for (x,y,w,h) in eye_l:
        #cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
        bound_eye.append([x,y,w,h])
    for (x,y,w,h) in nose:
        #cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
        bound_nose.append([x,y,w,h])
    
    print(bound_eye)
    print(bound_nose)
    #bound_eye = []
    
    img1 = draw_glasses(img1,bound_eye)
    img1 = draw_mustache(img1,bound_nose)
    
    return img1


def show_img(img1):
    cv2.imwrite('./Data/Test/After.png',img1)
    while True:
        cv2.imshow('img',img1)
        #cv2.imshow('glasses',img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    
    nd = img1.reshape((img1.shape[0]*img1.shape[1] , 3))
    print(nd.shape)
    nd = nd.tolist()
    try:
        os.remove('./Data/Test/output.csv')
    except():
        pass
    file = open('./Data/Test/output.csv','w')
    wr = csv.writer(file)
    wr.writerow(['b','g','r'])
    wr.writerows(nd)
    file.close()

def video_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose):
    cap = cv2.VideoCapture(0)
    cap.set(3, 720)
    cap.set(4, 480)
    
    while True:
        ret, img = cap.read()
        img = image_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose,img)   
        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()



    
video_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose)
#img1 = cv2.imread('Data/Test/Before.jpg')
#img1 = image_(face_cascade_eye_right,face_cascade_eye_left,face_cascade_nose,img1)
#show_img(img1)
