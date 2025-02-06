#This script is to go through all the images taken for rectification and select which are good
#The images will then be split into left and right images respectively 

import cv2 
import os
from user_settings import total_photos, img_height, img_width, rows, columns, square_size
# Global variables preset
#total_photos = 30
photo_height = 360
photo_width = 1280
#img_height = 360
#img_width = 640


# this script mode
DETECT_CHESSBOARD = True
# if True: not only show pictires but also try detect the chessboard and show its corners.
# DETECT_CHESSBOARD = False


if DETECT_CHESSBOARD:
    import numpy as np
    import json
    from stereovision.calibration import StereoCalibrator
    from stereovision.calibration import StereoCalibration
    from stereovision.exceptions import ChessboardNotFoundError


if DETECT_CHESSBOARD:
    image_size = (img_width, img_height)

    # This is the calibration class from the StereoVision package
    calibrator = StereoCalibrator(rows, columns, square_size, image_size)


def SeparateImages():
    if (os.path.isdir("../pairs") == False):
        os.makedirs("../pairs")

    for photo_counter in range(1, total_photos + 1):
        k = None
        filename = '../images/image_' + str(photo_counter).zfill(2) + '.png'
        if os.path.isfile(filename) == False:
            print("No file named " + filename)
            continue

        print("Image Pair: " + str(photo_counter))
        pair_img = cv2.imread(filename, -1)

        if DETECT_CHESSBOARD:
            imgLeft = pair_img[0:img_height, 0:img_width].copy()  # Y+H and X+W
            imgRight = pair_img[0:img_height, img_width:photo_width]

            # copy & paste of the esssential part of `3_calibration.py` ...

            # Ensuring both left and right images have the same dimensions
            (H, W, C) = imgLeft.shape

            imgRight = cv2.resize(imgRight, (W, H))

            # Calibrating the camera (getting the corners and drawing them)
            try:
                calibrator._get_corners(imgLeft)
                calibrator._get_corners(imgRight)
            except ChessboardNotFoundError as error:
                print(error)
                print("Pair No " + str(photo_counter) + " ignored: chessboard not found")
                continue
            else:
                calibrator.add_corners((imgLeft, imgRight), show_results=True)

        else:
            ...

        cv2.imshow("ImagePair: press `Y` if this one looks good.", pair_img)

        # waits for any key to be pressed
        k = cv2.waitKey(0) & 0xFF

        # print(' <> k: ', k, chr(k), ' | y={}, q={}'.format(ord('y'), ord('q')))

        if k == ord('y'):
            # save the photo
            imgLeft = pair_img[0:img_height, 0:img_width]  # Y+H and X+W
            imgRight = pair_img[0:img_height, img_width:photo_width]
            leftName = '../pairs/left_' + str(photo_counter).zfill(2) + '.png'
            rightName = '../pairs/right_' + str(photo_counter).zfill(2) + '.png'
            cv2.imwrite(leftName, imgLeft)
            cv2.imwrite(rightName, imgRight)
            print('+ + + Pair No ' + str(photo_counter) + ' saved.')

        elif k == ord('q'):
            break

        else:  # e.g.  k == ord('n')
            # skip the photo
            print("- - - Skipped.")

    print('End of cycle')
    
if __name__ == '__main__':

    print ("The paired images will be shown")
    print ("Press Y to accept & save the image")
    print("Press N to skip the image if it is blurry/unclear/cut-off or has poor chessboard detection")
    SeperateImages()



