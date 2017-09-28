
import cv2
import cv2 as cv
from os import listdir
import os
import time


def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path+'created successfully')


def cropImage(img, box, name):
    [p, q, r, s] = box
    # crop and save the image provided with the co-ordinates of bounding box
    write_img_color = img[q:q + s, p:p + r]
    saveCropped(write_img_color, name)


# save the cropped image at specified location
def saveCropped(img, name):
    cv2.imwrite(output_path + name + ".jpg", img)


if __name__ == "__main__":
    # paths to input and output images
    input_path = "newactors/"
    output_path = "newAllActors/"

    # load pre-trained frontalface cascade classifier
    frontal_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    print("Starting to detect faces in images and save the cropped images to output file...")
    sttime = time.clock()
    i = 1

    for subdir in listdir(input_path):
        paths=input_path+subdir+"/"
        input_names = listdir(input_path+subdir)


        output_path= "newAllActors/"+subdir+"/"

        print(output_path)
        createDir(output_path)

        for name in input_names:
            print(paths + name)
            color_img = cv2.imread(paths + name)
            # converting color image to grayscale image
            try:
                gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            except:
                print('could not convert to gray, skipping')
                continue
            # find the bounding boxes around detected faces in images
            bBoxes = frontal_face.detectMultiScale(gray_img, 1.3, 5)

            for box in bBoxes:
                # print(box)
                # crop and save the image at specified location
                cropImage(color_img, box, name)
                i += 1

print("Successfully completed the task in %.2f Secs." % (time.clock() - sttime))
