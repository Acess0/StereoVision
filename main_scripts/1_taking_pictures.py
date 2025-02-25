import cv2
import numpy as np
from start_cameras import Start_Cameras
from datetime import datetime
import time
import os
from os import path
from user_settings import total_photos, countdown


font = cv2.FONT_HERSHEY_SIMPLEX  # Font do countdown_timer


def TakePictures():
    val = input("Would you like to start the image capturing? (Y/N) ")
# Wcisniecie y zainicjalizuje kamery, pokaz podglad i zacznie odliczanie do zrobienia zdjecia.
    if val.lower() == "y":
        left_camera = Start_Cameras(0).start()
        right_camera = Start_Cameras(1).start()
        cv2.namedWindow("Images", cv2.WINDOW_NORMAL)

        counter = 0
        t2 = datetime.now()
        # Petla przechwytywania zdjec
        while counter <= total_photos:
            # Ustawianie odliczania 
            t1 = datetime.now()
            countdown_timer = countdown - int((t1 - t2).total_seconds())

            left_grabbed, left_frame = left_camera.read()
            right_grabbed, right_frame = right_camera.read()

            if left_grabbed and right_grabbed:
                # Łączenie ramek
                images = np.hstack((left_frame, right_frame))
                # Gdy skończy się odliczanie zegara “countdown”:
                # inkrementacja licznika zdjęć,
                # zapisanie obrazu w katalogu ../images/
                if countdown_timer == -1:
                    counter += 1
                    print(counter)

                 # Sprawdź czy istnieje katalog. Zapisz zdjęcie jeśli istnieje, utwórz folder i zapisz zdjęcie jeśli nie ma utworzonego katalogu
                    if path.isdir('../images') == True:
                        filename = "../images/image_" + str(counter).zfill(2) + ".png"
                        cv2.imwrite(filename, images)
                        print("Image: " + filename + " is saved!")
                    else:
                        # Tworzenie katalogu
                        os.makedirs("../images")
                        filename = "../images/image_" + str(counter).zfill(2) + ".png"
                        cv2.imwrite(filename, images)
                        print("Image: " + filename + " is saved!")

                    t2 = datetime.now()
                    # Wstrzymaj działanie na 1 sek
                    time.sleep(1)
                    countdown_timer = 0
                    next
                # Wyświetlanie odliczania i pokazywanie zdjęć    
                cv2.putText(images, str(countdown_timer), (50, 50), font, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imshow("Images", images)

                k = cv2.waitKey(1) & 0xFF

                if k == ord('q'):
                    break
                    
            else:
                break

    elif val.lower() == "n":
        print("Quitting! ")
        exit()
    else:
        print ("Please try again! ")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    TakePictures()
                
                
                
