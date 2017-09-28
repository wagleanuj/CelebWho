import cv2
import numpy as np
from os import listdir
import sys, time


def cropImage(img, box, name):
    [p, q, r, s] = box
    # crop and save the image provided with the co-ordinates of bounding box
    write_img_color = img[q:q + s, p:p + r]
    saveCropped(write_img_color, name)


# save the cropped image at specified location
def saveCropped(img, name):
    cv2.imwrite(output_path + name + ".jpg", img)


def get_images(path, size):
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
        for image in listdir(path + "/" + subdir):
            # print(subdir, images)
            img = cv2.imread(path + "/" + subdir + "/" + image, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, size)

            images.append(np.asarray(img, dtype=np.uint8))
            labels.append(sub)

            # cv2.waitKey(10)

        people.append(subdir)
        sub += 1
        print(people)
    return [images, labels, people]


if __name__ == "__main__":

    inputfiles = 'allactors/'
    outputfiles = 'tests'
    [images, labels, people] = get_images(inputfiles, (256, 256))
    # print([images, labels])

    labels = np.asarray(labels, dtype=np.int32)

    # initializing eigen_model and training
    print("Initializing eigen FaceRecognizer and training...")
    sttime = time.clock()
    eigen_model = cv2.face.createLBPHFaceRecognizer()
    eigen_model.train(images, labels)
    print("\tSuccessfully completed training in " + str(time.clock() - sttime) + " Secs!")
    eigen_model.save('savesinnew.yaml')

    # starting to predict subject/ person
    '''for image_name in listdir(outputfiles):
        try:
            pre_image = cv2.imread(outputfiles + "/" + image_name, cv2.IMREAD_GRAYSCALE)
            pre_image = cv2.resize(pre_image, (256, 256))
            frontal_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

            bBoxes = frontal_face.detectMultiScale(pre_image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))

            for bBox in bBoxes:
                (p, q, r, s) = bBox
                cv2.rectangle(pre_image, (p, q), (p + r, q + s), (225, 0, 25), 2)
                crop_gray_frame = pre_image[q:q + s, p:p + r]
                crop_gray_frame = cv2.resize(crop_gray_frame, (650, 490))
                n = []
                flag = 1
                predicted_label = eigen_model.predict(np.asarray(crop_gray_frame))
                print("Predicted person in the image " + image_name + " : " + people[predicted_label])

        except:
            print("Couldn't read image. Please check the path to image file.")
            sys.exit()

    eigen_model.save('saves')
    '''
