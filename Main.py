#Import thư viện

import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import DetectChars
import DetectPlates
import PossiblePlate

#Tùy chỉnh màu để nhận diện các hình ảnh vật thể
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = True  #Bật True để hiện và sử dụng các bước trong DetectChars và DetectPlates

##########################################################################
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # xây dựng theo mô hình KNN

    if blnKNNTrainingSuccessful == False:                               # Nếu Mô hình KNN sai thì báo lỗi
        print("\nerror: KNN traning was not successful\n") 
        return                                                       
    # end if

    imgOriginalScene  = cv2.imread("input/2.jpg")               #Địa chỉ nguồn mở ảnh nhận diện
    if imgOriginalScene is None:                            # nếu ảnh bị trống thì sẽ báo lỗi
        print("\nerror: image not read from file \n\n")  
        os.system("pause")                            
        return                              
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

    cv2.imshow("HIEN ANH GOC", imgOriginalScene)            #Hiện ảnh gốc

    if len(listOfPossiblePlates) == 0:                          #Nếu Plates không timg thấy
        print("\nno license plates were detected\n")  # không thể sử dụng Plates
    else:                                                         
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        licPlate = listOfPossiblePlates[0]

        cv2.imshow("CROP PHAN BIEN SO XE", licPlate.imgPlate)           # Hiện các ảnh cắt từ nhận diện của biển số
        cv2.imshow("BIEN SO DA DUOC NHAN DIEN", licPlate.imgThresh)
        cv2.imwrite("output/CROP PHAN BIEN SO XE.png",licPlate.imgPlate)
        cv2.imwrite("output/BIEN SO DA DUOC NHAN DIEN.png",licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # Nếu không tìm thấy Char
            print("\nno characters were detected\n\n")  # Hiện thông báo
            return                             
        # end if

        drawGreenRectangleAroundPlate(imgOriginalScene, licPlate)             #Vẽ vào hình ảnh nhận diện biển số theo màu sắc xanh

        cv2.imshow("ANH GOC VA DANH DAU PHAN NHAN DIEN", imgOriginalScene)               #Gọi lại ảnh gốc sau khi đã đánh dấu phần nhận diện
        cv2.imwrite("output/ANH GOC NHAN DIEN.png", imgOriginalScene)           #Tạo mục ảnh đã nhận diện vào thư mục output

    # end if else

    cv2.waitKey(0)					#kết thúc chương trình opencv

    return
# end main

###################################################################################################
def drawGreenRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # Dòng lệnh màu để đánh dấu trong ảnh nhận diện

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_GREEN, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_GREEN, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_GREEN, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_GREEN, 2)
# end function

if __name__ == "__main__":
    main()
