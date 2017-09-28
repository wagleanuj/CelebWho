import cv2


import numpy as np
from os import listdir
import sys, time
import os
import cv2
from PIL import Image

def cropImage(img, box,name):
	[p, q, r, s]= box
	# crop and save the image provided with the co-ordinates of bounding box
	write_img_color= img[q:q+ s, p:p+ r]
	saveCropped(write_img_color, name)

# save the cropped image at specified location
def saveCropped(img, name):
	cv2.imwrite(output_path+ name+ ".jpg", img)

def get_images(path):
    '''
    path: path to a folder which contains subfolders of for each subject/person
        which in turn cotains pictures of subjects/persons.

    size: a tuple to resize images.
        Ex- (256, 256)
    '''
    sub = 0
    images, labels = [], []
    people = []

    for subdir in listdir(path):

        people.append(subdir)
        sub += 1

    return  people

if __name__ == '__main__':
    eigenmodel=cv2.face.createLBPHFaceRecognizer()
    print('Loading from the existing data file.....')
    eigenmodel.load('savesinnew.yaml')
    testfiles='tests'
    people=['ALISHA_RAI', 'ALISHA_SHARMA', 'AMISHA_BASNET', 'ANMOL_KC', 'ANNA_SHARMA', 'ANUP_BARAL', 'ARPAN_THAPA', 'ARUNIMA_LAMSAL', 'ARYAN_SIGDEL', 'ASHISHMA_NAKARMI', 'BHUWAN_KC', 'BINITA_BARAL', 'BIPANA_THAPA', 'BIRAJ_BHATTA', 'DAYAHANG_RAI', 'DEEPAK_RAJ_GIRI', 'DICHEN_LACHMAN', 'DILIP_RAYAMAJHI', 'GARIMA_PANTA', 'GAURI_MALLA', 'HARI_BANSA_ACHARYA', 'JAL_SHAH', 'JHARNA_THAPA', 'JIWAN_LUITEL', 'KARISHMA_MANANDHAR', 'KEKI_ADHIKARI', 'MADAN_KRISHNA_SHRESHTHA', 'MAHIMA_SILWAL', 'MALINA_JOSHI', 'MALVIKA_SUBBA', 'MELINA_MANANDHAR', 'NAJIR_HUSSAIN', 'NAMRATA_SHRESTHA', 'NANDITA_KC', 'NEER_SHAH', 'NIKHIL_UPRETI', 'NIMA_RUMBA', 'NIRUTA_SINGH', 'NISHA_ADHIKARI', 'NISHCHAL_BASNET', 'PAUL_SHAH', 'PRAKREETI_SHRESTHA', 'PRIYANKA_KARKI', 'RAJESH_HAMAL', 'RAJ_BALLAV_KOIRALA', 'REECHA_SHARMA', 'REKHA_THAPA', 'RISHMA_GURUNG', 'SAMRAGYEE_RL_SHAH', 'SANCHITA_LUITEL', 'SAUGAT_MALLA', 'SHIV_SHRESHTHA', 'SHRISTI_SHRESTHA', 'SIPORA_GURUNG', 'SITARAM_KATTEL', 'SRI_KRISHNA_SHRESHTHA', 'SUMI_KHADKA', 'SUNIL_RAWAL', 'SUNIL_THAPA', 'SURAJ_SINGH', 'SUSHIL_CHHETRI', 'SUSHMA_KARKI', 'SWASTIMA_KHADKA', 'TRIPTI_NADAKAR', 'TULSI_GHIMIRE', 'UDIT_NARAYAN_JHA', 'USHA_POUDEL', 'USHA_RAJAK', 'WILSON_BIKRAM_RAI', 'ZENISHA_MOKTAN']
    # starting to predict subject/ person
    for image_name in listdir(testfiles):
        try:
            pre_image = cv2.imread(testfiles + "/" + image_name, cv2.IMREAD_GRAYSCALE)
            print(testfiles+"/"+image_name)
        except:
            print("Couldn't read image. Please check the path to image file.")
            sys.exit()
        #pre_image = cv2.resize(pre_image, (256, 256))
        frontal_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        bBoxes = frontal_face.detectMultiScale(pre_image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))

        for bBox in bBoxes:
            (p, q, r, s) = bBox
            cv2.rectangle(pre_image, (p, q), (p + r, q + s), (225, 0, 25), 2)
            crop_gray_frame = pre_image[q:q + s, p:p + r]
            crop_gray_frame = cv2.resize(crop_gray_frame, (256, 256))
            n = []
            flag = 1

            predicted_label = eigenmodel.predict(np.asarray(crop_gray_frame))
            print("This person in the image ' "+image_name+"' looks like" + ":"+ people[predicted_label[0]]+" Confidence level: "+str(predicted_label[1]) )
            ff="/media/wagleanuj/UNTITLED/hp spectre/New folder (2)/FaceRecognizer-Wagle/allactors/frame.jpg"
            filename=(people[predicted_label[0]])
            newpath=os.path.join("allactors",filename)
            g=listdir(newpath)[0]
            ss=os.path.join(newpath,g)


            img=Image.open(ss)
            img.show()



