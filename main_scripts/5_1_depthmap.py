import cv2
import numpy as np
import os
import json
from stereovision.calibration import StereoCalibration
from start_cameras import Start_Cameras
from os import path
SWS = 5      # SADWindowSize
PFS = 5      # preFilterSize
PFC = 29     # preFilterCap
MDS = -30    # minDisparity
NOD = 160    # numberOfDisparities
TTH = 100    # textureThreshold
UR = 10      # uniquenessRatio
SR = 14      # speckleRange
SPWS = 100   # speckleWindowSize

def load_map_settings(file):

    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, sbm

    print('Loading parameters from file...')

    with open(file, 'r') as f:

        data = json.load(f)
        SWS = data['SADWindowSize']
        PFS = data['preFilterSize']
        PFC = data['preFilterCap']
        MDS = data['minDisparity']
        NOD = data['numberOfDisparities']
        TTH = data['textureThreshold']
        UR = data['uniquenessRatio']
        SR = data['speckleRange']
        SPWS = data['speckleWindowSize']

    # Inicjalizacja obiektu StereoBM
    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=SWS) 
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)

    print('Parameters loaded from file ' + file)

    return sbm


def stereo_depth_map(rectified_pair):

    # Zrektyfikowane obrazy
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]

    # Oblicz  mapę dysparycji; StereoBM zwraca CV_16S, dlatego konwertujemy do float32
    disparity = sbm.compute(dmLeft, dmRight).astype(np.float32) / 17.0

    # Dla wizualizacji – normalizujemy do zakresu 0-255
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_normalized = np.array(disparity_normalized, dtype=np.uint8)
    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

    # Zwracamy mapę kolorową do wyświetlania oraz surową mapę dysparycji do obliczeń
    return disparity_color, disparity


def onMouse(event, x, y, flags, param):
    # 'param' będzie zawierać surową mapę dysparycji
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


if __name__ == "__main__":
    # Załaduj ustawienia mapy dysparycjii i parametry kalibracyjne:
    sbm = load_map_settings("../3dmap_set.txt")
    cam_mat_left = np.load("../calib_result/cam_mats_left.npy")
    focal_length = cam_mat_left[0, 0]
    R = np.load("../calib_result/rot_mat.npy")
    T = np.load("../calib_result/trans_vec.npy")
    
    # Ogniskowa jest w pliku "cam_mats_left.npy", a odleglosc miedzy kamerami w "trans_vec.npy"
    focal_length = cam_mat_left[0, 0]  # przyjmujemy focal_length jako element [0,0]
    baseline = np.linalg.norm(T)  # baseline w centymetrach
    print("Focal length: {:.2f} px, Baseline: {:.2f}".format(focal_length, baseline))

    print ("Press S to save depth map")
    
    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()

    cv2.namedWindow("DepthMap")

    while True:
        left_grabbed, left_frame = left_camera.read()
        right_grabbed, right_frame = right_camera.read()

        if left_grabbed and right_grabbed:  
            # Konwertuj obrazy do odcieni szarości
            left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)


            # Użyj wyników kalibracji do rektyfikacji
            calibration = StereoCalibration(input_folder='../calib_result')
            rectified_pair = calibration.rectify((left_gray_frame, right_gray_frame))

            # Wygeneruj mape dysparycji i głębi na podstawie zrektyfikowanej pary obrazów
            disparity_color, disparity = stereo_depth_map(rectified_pair)
            depth_matrix = np.where(disparity > 0, (focal_length * baseline) / disparity, 0)
            
            # Ustaw callback myszy, przekazując surową mapę dysparycji (disparity) jako parametr
            cv2.setMouseCallback("DepthMap", onMouse, disparity)
    
            # Nałożenie mapy dysparycji na oryginalny obraz dla wizualizacji
            if left_frame.shape != disparity_color.shape:
             disparity_color = cv2.resize(disparity_color, (left_frame.shape[1], left_frame.shape[0]))
            output = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)

            cv2.imshow("DepthMap", np.hstack((disparity_color, output)))																																																																							

    

            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):

                break

            elif k == ord('s'):
             #cv2.imwrite("../maps/disparity_matrix.png", disparity)
             #cv2.imwrite("../disparity/depth_matrix.png", depth_matrix)
             # Zapisywanie mapy głębi w formacie .png do katalogu ../depth/
                if path.isdir('../depth') == True:
                        filename = "../depth/depth_map"  + ".png"
                        cv2.imwrite(filename, depth_matrix)
                        print("Depth map: " + filename + " is saved!")
                else:
                        # Tworzenie katalogu
                        os.makedirs("../depth")
                        filename = "../depth/depth_map"  + ".png"
                        cv2.imwrite(filename, depth_matrix)
                        print("Depth map: " + filename + " is saved!")
            continue
    cv2.destroyAllWindows()
