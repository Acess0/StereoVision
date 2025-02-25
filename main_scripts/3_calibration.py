import os
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError
from user_settings import total_photos, img_height, img_width, rows, columns, square_size


# Pokaz wykryte punkty charakterystyczne (True, False) 
PREVIEW_DETECTION_RESULTS = True



image_size = (img_width, img_height)
# Klasa calibration z pakietu StereoVision
calibrator = StereoCalibrator(rows, columns, square_size, image_size)
photo_counter = 0
print('Start cycle')

# Przejdź przez każdą parę zdjęć
while photo_counter != total_photos:
    photo_counter += 1 
    print('Importing pair: ' + str(photo_counter))
    leftName = '../pairs/left_' + str(photo_counter).zfill(2) + '.png'
    rightName = '../pairs/right_' + str(photo_counter).zfill(2) + '.png'
    if os.path.isfile(leftName) and os.path.isfile(rightName):
        imgLeft = cv2.imread(leftName, 1)
        imgRight = cv2.imread(rightName, 1)
        # Sprawdzanie czy wymiary zdjęcia z kamery lewej = zdjęcia kamerze prawej
        (H, W, C) = imgLeft.shape

        imgRight = cv2.resize(imgRight, (W, H))
        
        # Kalibracja kamer (wyznaczanie punktów charakterystycznych i ich rysowanie)
        try:
            calibrator._get_corners(imgLeft)
            calibrator._get_corners(imgRight)
        except ChessboardNotFoundError as error:
            print(error)
            print("Pair No " + str(photo_counter) + " ignored")
        else:
            ## definition: add_corners(self, image_pair, show_results=False)
            calibrator.add_corners((imgLeft, imgRight), show_results=PREVIEW_DETECTION_RESULTS)
        
    else:
        print ("Pair not found")
        continue


print('Cycle Complete!')

print('Starting calibration... It can take several minutes!')
calibration = calibrator.calibrate_cameras()
calibration.export('../calib_result')
print('Calibration complete!')

# Zrektyfikowanie pary obrazów i jej pokazanie
calibration = StereoCalibration(input_folder='../calib_result')
rectified_pair = calibration.rectify((imgLeft, imgRight))

cv2.imshow('Left Calibrated!', rectified_pair[0])
cv2.imshow('Right Calibrated!', rectified_pair[1])
cv2.imwrite("../rectified_left.jpg", rectified_pair[0])
cv2.imwrite("../rectified_right.jpg", rectified_pair[1])
cv2.waitKey(0)
