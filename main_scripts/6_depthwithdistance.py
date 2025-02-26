# Wyznaczanie dystansu do wykrytego obiektu
import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration
from start_cameras import Start_Cameras
import jetson_inference
import jetson_utils

# Domyślne parametry StereoBM
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 160
TTH = 100
UR = 10
SR = 14
SPWS = 100


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

    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight).astype(np.float32) / 16.0
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_normalized = np.array(disparity_normalized, dtype=np.uint8)
    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)


    return disparity_color, disparity
    

def onMouse(event, x, y, flags, param):
    disparity = param
    if event == cv2.EVENT_LBUTTONDOWN:
        d =  disparity[y, x]
        if d <= 0:
            print("Disparity is 0 or invalid at this point, cannot compute distance")
        else:
            # Oblicz dystans: (focal_length * baseline) / disparity
            dist = (focal_length * baseline) / d
            print("Distance: {:.1f} centimeters".format(dist))
        return dist
        
        
# Wczytanie etykiet COCO
with open('../SSD-Mobilenet-v2/ssd_coco_labels.txt', 'r') as f:
    coco_labels = [line.strip() for line in f.readlines()]

# Model detekcji objetków
net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

if __name__ == "__main__":

# Załaduj kalibracyjne parametry: focal_length i wektor translacji
# Zakładamy, że focal_length jest w pliku "cam_mats_left.npy" i wektor translacji w "trans_vec.npy"
    cam_mat_left = np.load("../calib_result/cam_mats_left.npy")
    focal_length = cam_mat_left[0, 0]  # przyjmujemy focal_length jako element [0,0]
    trans_vec = np.load("../calib_result/trans_vec.npy")
    baseline = np.linalg.norm(trans_vec)  # baseline w centymetrach
    #print("Focal length: {:.2f} px, Baseline: {:.2f}".format(focal_length, baseline))
    
    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()
    load_map_settings("../3dmap_set.txt")
    cv2.namedWindow("DepthMap")

    try:
        while True:
            left_grabbed, left_frame = left_camera.read()
            right_grabbed, right_frame = right_camera.read()

            if left_grabbed and right_grabbed:
                left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                calibration = StereoCalibration(input_folder='../calib_result')
                rectified_pair = calibration.rectify((left_gray, right_gray))
                disparity_color, disparity = stereo_depth_map(rectified_pair)
                disparity_color = cv2.resize(disparity_color, (left_frame.shape[1], left_frame.shape[0]))
                disparity = cv2.resize(disparity, (left_frame.shape[1], left_frame.shape[0]))

                cv2.setMouseCallback("DepthMap", onMouse, disparity)

                # Wykonaj detekcje obiektów
                # Kopiowanie macierzy numpy do pamięci CUDA
                left_cuda = jetson_utils.cudaFromNumpy(left_frame) 
                    #detections = net.Detect(img, overlay=opt.overlay)
                    #https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet console-2.md
                    #Przeprowadzanie detekcji w rdzeniach CUDA
                detections = net.Detect(left_cuda, overlay="box,labels,conf") 

                # Przetwórz obiekty
                for detection in detections:
                    class_id = detection.ClassID
                    if class_id >= len(coco_labels):
                        continue  # Pomin nieprawidłowe ID

                    # Uzyskanie koordynatów obiektu
                    left = int(detection.Left)
                    top = int(detection.Top)
                    right = int(detection.Right)
                    bottom = int(detection.Bottom)
                    center_x = int(detection.Center[0])
                    center_y = int(detection.Center[1])
                    
                    # Oblicz dystans i pobierz klase
                    d = disparity[center_y, center_x]
                    distance = (focal_length * baseline) / d
                    label_name = coco_labels[class_id]

                    # Narysuj bounding box i wypisz etykiete
                    cv2.rectangle(left_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    label_text = f"{label_name}: {distance:.1f}cm"
                    cv2.putText(left_frame, label_text, (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Wyświetl przetworzony obraz
                left_stacked = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0)
                cv2.imshow("DepthMap", np.hstack((disparity_color, left_stacked)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:

        cv2.destroyAllWindows()
