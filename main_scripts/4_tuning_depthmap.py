import cv2
import os
import threading
import numpy as np
import time
from datetime import datetime
import json
from stereovision.calibration import StereoCalibration
from start_cameras import Start_Cameras

SWS = 215    # SADWindowSize
PFS = 115    # preFilterSize
PFC = 43     # preFilterCap
MDS = -25    # minDisparity
NOD = 112    # numberOfDisparities
TTH = 100    # textureThreshold
UR = 10      # uniquenessRatio
SR = 15      # speckleRange
SPWS = 100   # speckleWindowSize

loading = False


def stereo_depth_map(rectified_pair, variable_mapping):

    # Ustawienia parametrów
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=variable_mapping["SWS"]) 
    sbm.setPreFilterType(1)    
    sbm.setPreFilterSize(variable_mapping['PreFiltSize'])
    sbm.setPreFilterCap(variable_mapping['PreFiltCap'])
    sbm.setSpeckleRange(variable_mapping['SpeckleRange'])
    sbm.setSpeckleWindowSize(variable_mapping['SpeckleSize'])
    sbm.setMinDisparity(variable_mapping['MinDisp'])
    sbm.setNumDisparities(variable_mapping['NumofDisp'])
    sbm.setTextureThreshold(variable_mapping['TxtrThrshld'])
    sbm.setUniquenessRatio(variable_mapping['UniqRatio'])
    
    # Wczytywanie obrazów po rektyfikacji
    c, r = rectified_pair[0].shape
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    # Obliczanie mapy dysparycji
    disparity = sbm.compute(dmLeft, dmRight).astype(np.float32) / 23.0
    # Dla wizualizacji – normalizujemy do zakresu 0-255
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_normalized = np.array(disparity_normalized, dtype=np.uint8)
    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    # Zwracamy mapę kolorową do wyświetlania oraz mapę dysparycji do obliczeń
    return disparity_color, disparity


def save_load_map_settings(current_save, current_load, variable_mapping):
    global loading
    if current_save != 0:
        print('Saving to file...')

        result = json.dumps({'SADWindowSize':variable_mapping["SWS"], 'preFilterSize':variable_mapping['PreFiltSize'], 'preFilterCap':variable_mapping['PreFiltCap'], 
                'minDisparity':variable_mapping['MinDisp'], 'numberOfDisparities': variable_mapping['NumofDisp'], 'textureThreshold':variable_mapping['TxtrThrshld'], 
                'uniquenessRatio':variable_mapping['UniqRatio'], 'speckleRange':variable_mapping['SpeckleRange'], 'speckleWindowSize':variable_mapping['SpeckleSize']},
                sort_keys=True, indent=4, separators=(',',':'))
        fName = '../3dmap_set.txt'
        f = open(str(fName), 'w')
        f.write(result)
        f.close()
        print ('Settings saved to file '+fName)


    if current_load != 0:
        if os.path.isfile('../3dmap_set.txt') == True:
            loading = True
            fName = '../3dmap_set.txt'
            print('Loading parameters from file...')
            f = open(fName, 'r')
            data = json.load(f)

            cv2.setTrackbarPos("SWS", "Stereo", data['SADWindowSize'])
            cv2.setTrackbarPos("PreFiltSize", "Stereo", data['preFilterSize'])
            cv2.setTrackbarPos("PreFiltCap", "Stereo", data['preFilterCap'])
            cv2.setTrackbarPos("MinDisp", "Stereo", data['minDisparity']+100)
            cv2.setTrackbarPos("NumofDisp", "Stereo", int(data['numberOfDisparities']/16))
            cv2.setTrackbarPos("TxtrThrshld", "Stereo", data['textureThreshold'])
            cv2.setTrackbarPos("UniqRatio", "Stereo", data['uniquenessRatio'])
            cv2.setTrackbarPos("SpeckleRange", "Stereo", data['speckleRange'])
            cv2.setTrackbarPos("SpeckleSize", "Stereo", data['speckleWindowSize'])

            f.close()
            print ('Parameters loaded from file '+fName)
            print ('Redrawing depth map with loaded parameters...')
            print ('Done!') 

        else: 
            print ("File to load from doesn't exist.")

def activateTrackbars(x):
    global loading
    loading = False
    

def create_trackbars() :
    global loading

    cv2.createTrackbar("SWS", "Stereo", 115, 230, activateTrackbars)
    cv2.createTrackbar("SpeckleSize", "Stereo", 0, 300, activateTrackbars)
    cv2.createTrackbar("SpeckleRange", "Stereo", 0, 40, activateTrackbars)
    cv2.createTrackbar("UniqRatio", "Stereo", 1, 20, activateTrackbars)
    cv2.createTrackbar("TxtrThrshld", "Stereo", 0, 1000, activateTrackbars)
    cv2.createTrackbar("NumofDisp", "Stereo", 1, 16, activateTrackbars)
    cv2.createTrackbar("MinDisp", "Stereo", -100, 200, activateTrackbars)
    cv2.createTrackbar("PreFiltCap", "Stereo", 1, 63, activateTrackbars)
    cv2.createTrackbar("PreFiltSize", "Stereo", 5, 255, activateTrackbars)
    cv2.createTrackbar("Save Settings", "Stereo", 0, 1, activateTrackbars)
    cv2.createTrackbar("Load Settings","Stereo", 0, 1, activateTrackbars)

def onMouse(event, x, y, flags, param):
    disparity = param
    if event == cv2.EVENT_LBUTTONDOWN:
        d = disparity[y, x]
        if d <= 0:
            print("Disparity is 0 or invalid at this point, cannot compute distance")
        else:
            # Oblicz dystans: (focal_length * baseline) / disparity
            dist = (focal_length * baseline) / d
            print("Distance: {:.1f} centimeters".format(dist))
        return dist


if __name__ == '__main__':
    cam_mat_left = np.load("../calib_result/cam_mats_left.npy")
    focal_length = cam_mat_left[0, 0]  # przyjmujemy focal_length jako element [0,0]
    T = np.load("../calib_result/trans_vec.npy")
    baseline = np.linalg.norm(T)  # baseline w centymetrach
    #print("Focal length: {:.2f} px, Baseline: {:.2f}".format(focal_length, baseline))
    
    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()

    # Inicjalizacja suwaków i okien
    cv2.namedWindow("Stereo")
    create_trackbars()

    print ("Cameras Started")

    variables = ["SWS", "SpeckleSize", "SpeckleRange", "UniqRatio", "TxtrThrshld", "NumofDisp",
    "MinDisp", "PreFiltCap", "PreFiltSize"]

    variable_mapping = {"SWS" : 15, "SpeckleSize" : 100, "SpeckleRange" : 15, "UniqRatio" : 10, "TxtrThrshld" : 100, "NumofDisp" : 1,
    "MinDisp": -25, "PreFiltCap" : 30, "PreFiltSize" : 105}

    while True:
        left_grabbed, left_frame = left_camera.read()
        right_grabbed, right_frame = right_camera.read()

        if left_grabbed and right_grabbed:
            left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            calibration = StereoCalibration(input_folder='../calib_result')
            rectified_pair = calibration.rectify((left_gray_frame, right_gray_frame))
            
            # Uzyskiwanie pozycji suwaków i przypisywanie ich wartości do zmiennych
            if loading == False:
                for v in variables:
                    current_value = cv2.getTrackbarPos(v, "Stereo")
                    if v == "SWS" or v == "PreFiltSize":
                        if current_value < 5:
                            current_value = 5
                        if current_value % 2 == 0:
                            current_value += 1
                    
                    if v == "NumofDisp":
                        if current_value == 0:
                            current_value = 1
                        current_value = current_value * 16
                    if v == "MinDisp":
                        current_value = current_value - 100
                    if v == "UniqRatio" or v == "PreFiltCap":
                        if current_value == 0:
                            current_value = 1
                    
                    variable_mapping[v] = current_value


            
           # Zapisywanie i odczytywanie parametrów mapy głębi
            current_save = cv2.getTrackbarPos("Save Settings", "Stereo")
            current_load = cv2.getTrackbarPos("Load Settings", "Stereo")
 
            save_load_map_settings(current_save, current_load, variable_mapping)
            cv2.setTrackbarPos("Save Settings", "Stereo", 0)
            cv2.setTrackbarPos("Load Settings", "Stereo", 0)
            disparity_color, disparity = stereo_depth_map(rectified_pair, variable_mapping)
            depth_matrix = np.where(disparity > 0, (focal_length * baseline) / disparity, 0)

            # Wywoływanie fukcji onMouse 
            cv2.setMouseCallback("Stereo", onMouse, disparity)
                      
            cv2.imshow("Stereo", disparity_color)
            cv2.imshow("Frame", np.hstack((rectified_pair[0], rectified_pair[1])))
            
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
            
              break

            continue

    cv2.destroyAllWindows()
                

