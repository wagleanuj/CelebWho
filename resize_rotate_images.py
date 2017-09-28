

import cv2
from os import listdir
import os

input_path = "input_images/"
output_path = "new_resize/"


def createDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(path + 'created successfully')


if __name__ == '__main__':

    images = listdir("" + input_path)
    for subdir in listdir('input_images'):
        images = listdir("" + input_path + subdir + "/")
        for image in images:
            img = cv2.imread(input_path + subdir + "/" + image)
            resize_img = cv2.resize(img, (650, 490))
            (h, w) = resize_img.shape[:2]
            center = (w / 2, h / 2)

            matrix = cv2.getRotationMatrix2D(center, 0, 0.8)
            rotated_img = cv2.warpAffine(resize_img, matrix, (w, h))
            createDir(output_path+subdir+"/")
            cv2.imwrite(output_path +subdir+"/"+ image, rotated_img)
            cv2.imshow("Win", rotated_img)
            cv2.waitKey(5)
