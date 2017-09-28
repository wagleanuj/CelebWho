'''
Surya Teja Cheedella
shine123surya[at]gmail[dot]com
BITS Pilani, Hyderabad Campus

    Real-Time detection & prediction of subjects/persons in
        video recording by in-built camera.
    If there is any intruder (trained/ unknown subjects) attack, it posts on your
        facebook timeline to notify you and your friends/ neighbours.

Working:
    Takes images stored in first path and traines faceRecognizer models.
    Then starts recording video from camera and shows detected subjects.

Usage:
    face_detrec_video.py <full/path/to/root/images/folder>

Takes one argument:
    1. Input folder which contains sub-folders of subjects/ persons.
        There should be images saved in subfolders which are used to train.
'''

import cv2
import cv2 as cv
import numpy as np
import os
import sys, time
from PIL import Image

def get_images(path, size):
    '''
    path: path to a folder which contains subfolders of for each subject/person
        which in turn cotains pictures of subjects/persons.

    size: a tuple to resize images.
        Ex- (256, 256)
    '''
    sub= 0
    images, labels= [], []
    people= []

    for subdir in os.listdir(path):
        for image in os.listdir(path+ "/"+ subdir):
            #print(subdir, images)
            img= cv2.imread(path+os.path.sep+subdir+os.path.sep+image, cv2.IMREAD_GRAYSCALE)
            img= cv2.resize(img, size)

            images.append(np.asarray(img, dtype= np.uint8))
            labels.append(sub)

            cv2.imshow("win", img)
            cv2.waitKey(10)

        people.append(subdir)
        sub+= 1

    return [images, labels, people]

def detect_faces(image):
    '''
    Takes an image as input and returns an array of bounding box(es).
    '''
    frontal_face= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    bBoxes= frontal_face.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))
    return bBoxes

def train_model(path):
    '''
    Takes path to images and train a face recognition model
    Returns trained model and people
    '''
    [images, labels, people]= get_images(arg_one, (256, 256))
    #print([images, labels])

    labels= np.asarray(labels, dtype= np.int32)

    # initializing eigen_model and training
    print("Initializing eigen FaceRecognizer and training...")
    sttime= time.clock()

    eigen_model= cv2.face.createLBPHFaceRecognizer(10,10)
    eigen_model.train(images, labels)
    print("\tSuccessfully completed training in "+ str(time.clock()- sttime)+ " Secs!")

    return [eigen_model, people]

def majority(mylist):
    '''
    Takes a list and returns an element which has highest frequency in the given list.
    '''
    myset= set(mylist)
    ans= mylist[0]
    ans_f= mylist.count(ans)

    for i in myset:
        if mylist.count(i)> ans_f:
            ans= i
            ans_f= mylist.count(i)

    return ans




if __name__== "__main__":
   # if len(sys.argv)!= 2:
    #    print("Wrong number of arguments! See the usage.\n")
     #   print("Usage: face_detrec_video.py <full/path/to/root/images/folder>")
      #  sys.exit()

    arg_one= "output_images"
    eigen_model, people= train_model(arg_one)

    #starts recording video from camera and detects & predict subjects
    cap= cv2.VideoCapture(0)

    counter= 0
    last_20= [0 for i in range(20)]
    final_5= []
    box_text= "Subject: "


    while(True):
        ret, frame= cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)

        bBoxes= detect_faces(gray_frame)

        for bBox in bBoxes:
            (p,q,r,s)= bBox
            cv2.rectangle(frame, (p,q), (p+r,q+s), (225,0,25), 2)

            crop_gray_frame= gray_frame[q:q+s, p:p+r]
            crop_gray_frame= cv2.resize(crop_gray_frame, (256, 256))

            predicted_label= eigen_model.predict(np.asarray(crop_gray_frame))

            last_20.append(predicted_label)
            last_20= last_20[1:]





            cv2.putText(frame, box_text, (p-20, q-5), cv2.FONT_HERSHEY_PLAIN, 1.3, (25,0,225), 2)

            if counter%10== 0:
                max_label= majority(last_20)
                #box_text= format("Subject: "+ people[max_label])
                box_text= format("Subject: "+ people[predicted_label])

                if counter> 20:
                    final_5.append(max_label)       #it always takes max_label into consideration
                    if len(final_5)== 5:
                        final_label= majority(final_5)

                        picture_name= "frame.jpg"
                        cv2.imwrite(picture_name, frame)
                        final_5= []




        cv2.imshow("Video Window", frame)
        counter+= 1

        if (cv2.waitKey(5) & 0xFF== 27):
            break

    cv2.destroyAllWindows()
