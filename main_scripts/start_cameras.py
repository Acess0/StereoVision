import cv2
import numpy as np
import depthai as dai
import threading
import time

class Start_Cameras:
    # Współdzielone zasoby dla wszystkich instancji
    _device = None
    _q_left = None
    _q_right = None
    _latest_frames = {0: None, 1: None}
    _capture_thread = None
    _lock = threading.Lock()

    def __init__(self, cam_id):
        """
        Konstruktor przyjmujący identyfikator kamery:
         0 - kamera lewa,
         1 - kamera prawa.
        """
        if cam_id not in [0, 1]:
            raise ValueError("Wrong camera ID.")
        self.cam_id = cam_id
        with Start_Cameras._lock:
            if Start_Cameras._device is None:
                self._init_device()

    @staticmethod
    def _init_device():
        pipeline = dai.Pipeline()

        # Konfiguracja lewej kamery mono
        cam_left = pipeline.create(dai.node.MonoCamera)
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

        # Konfiguracja prawej kamery mono
        cam_right = pipeline.create(dai.node.MonoCamera)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

        # Utworzenie wyjść XLink
        xout_left = pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName("left")
        cam_left.out.link(xout_left.input)

        xout_right = pipeline.create(dai.node.XLinkOut)
        xout_right.setStreamName("right")
        cam_right.out.link(xout_right.input)

        # Połączenie z urządzeniem i uzyskanie kolejki następnych ramek
        Start_Cameras._device = dai.Device(pipeline)
        Start_Cameras._q_left = Start_Cameras._device.getOutputQueue(name="left", maxSize=4, blocking=False)
        Start_Cameras._q_right = Start_Cameras._device.getOutputQueue(name="right", maxSize=4, blocking=False)

        # Uruchomienie wątku do ciągłego pobierania ramek
        Start_Cameras._capture_thread = threading.Thread(target=Start_Cameras._capture_loop, daemon=True)
        Start_Cameras._capture_thread.start()

    @staticmethod
    def _capture_loop():
        while True:
            # Pobranie ramki z lewej kamery przy użyciu tryGet()
            frame_left_packet = Start_Cameras._q_left.tryGet()
            if frame_left_packet is not None:
                frame_left = frame_left_packet.getFrame()
                frame_left = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2BGR)
                Start_Cameras._latest_frames[0] = frame_left

            # Pobranie ramki z prawej kamery przy użyciu tryGet()
            frame_right_packet = Start_Cameras._q_right.tryGet()
            if frame_right_packet is not None:
                frame_right = frame_right_packet.getFrame()
                frame_right = cv2.cvtColor(frame_right, cv2.COLOR_GRAY2BGR)
                Start_Cameras._latest_frames[1] = frame_right

 #           	time.sleep(0.01)

    def start(self):
        return self

    def read(self):
        """
        Zwraca krotkę (grabbed, frame):
          - grabbed: True, jeśli klatka jest dostępna, False w przeciwnym przypadku.
          - frame: obraz z kamery lub None.
        """
        frame = Start_Cameras._latest_frames[self.cam_id]
        if frame is None:
            return False, None
        return True, frame


if __name__ == "__main__":
    left_camera = Start_Cameras(0).start()
    right_camera = Start_Cameras(1).start()

    while True:
        left_grabbed, left_frame = left_camera.read()
        right_grabbed, right_frame = right_camera.read()

        # Jeśli klatki nie zostały jeszcze pobrane, pomiń iterację
        if not left_grabbed or not right_grabbed:
            continue

        combined = np.hstack((left_frame, right_frame))
        cv2.imshow("OAK-D Lite: Left + Right Cameras", combined)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

