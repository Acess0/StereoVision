import cv2 
import os
from user_settings import total_photos, img_height, img_width, photo_width, photo_height

def SeperateImages():
    photo_counter = 1
    
    if (os.path.isdir("../pairs") == False):
        os.makedirs("../pairs")
        
    while photo_counter != total_photos:
        k = None
        filename = '../images/image_'+ str(photo_counter).zfill(2) + '.png'
        if os.path.isfile(filename) == False:
            print("No file named " + filename)
            photo_counter += 1
            
            continue
        pair_img = cv2.imread(filename, -1)
        
        print ("Image Pair: " + str(photo_counter))
        cv2.imshow("ImagePair", pair_img)
        
        # Czekanie na klikniecie przycisku
        k = cv2.waitKey(0) & 0xFF
        # Rozbicie połączonego obrazu na obraz z kamery lewej i prawej
        if k == ord('y'):
            # Zapisz zdjęcie
            imgLeft = pair_img[0:img_height, 0:img_width]  # Y+H and X+W
            imgRight = pair_img[0:img_height, img_width:photo_width]
            # Przypisanie nazwy "left" i "right" + numer zdjęcia + .png w katalogu pairs
            leftName = '../pairs/left_' + str(photo_counter).zfill(2) + '.png'
            rightName = '../pairs/right_' + str(photo_counter).zfill(2) + '.png'
            cv2.imwrite(leftName, imgLeft)
            cv2.imwrite(rightName, imgRight)
            print('Pair No ' + str(photo_counter) + ' saved.')
            photo_counter += 1
     
        elif k == ord('n'):
            # Pomiń zdjęcie
            photo_counter += 1
            print ("Skipped")
            
        elif k == ord('q'):
            break  
  

            
    
    print('End cycle')
    
if __name__ == '__main__':

    print ("The paired images will be shown")
    print ("Press Y to accept & save the image")
    print ("Press N to skip the image if it is blurry/unclear/cut-off") 
    SeperateImages()



