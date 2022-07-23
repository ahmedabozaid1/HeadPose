import os
import cv2
import numpy as np
import cvlib as cv


 
paths = ["../Experiment/focus", "../Experiment/left", "../Experiment/right"]
names = ["focus","left","right"]


def save_faces(imgname,direction):   
    #print(os.path.join(image_path, imgname))
    #return
  
    img = cv2.imread(os.path.join(image_path, imgname))

     # apply face detection
    face, confidence = cv.detect_face(img)
    print(confidence)
    # loop through detected faces
    i=0
    for idx, f in enumerate(face):
        i=i+1
        print(i)
        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]


        # crop the detected face region
        face_crop = np.copy(img[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        fac, conf = cv.detect_face(face_crop)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        if i == 1:
            cv2.imwrite('../Experiment/proccessed/{}/{}'.format(direction, imgname), gray)
            print(imgname,direction ," saved")
    i=0
 

if __name__ == '__main__':


    for i in range(3):
        image_path = paths[i]
        direction=names[i]
    # Iterate through files
        for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:
            save_faces(f,direction)
     
