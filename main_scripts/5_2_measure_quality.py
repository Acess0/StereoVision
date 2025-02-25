import cv2
import numpy as np
import argparse
import math
def quality(image1_path, image2_path, bad_threshold=1.0):
    """
    Funkcja wczytuje dwa obrazy i oblicza RMS oraz procent pikseli przekraczających zadany próg.
    
    Parametry:
        image1_path (str): ścieżka do wyznaczonej mapy dysparycji.
        image2_path (str): ścieżka do wzorcowej mapy dysparycji.
        bad_threshold (float): próg błędu dla obliczenia % pikseli nietrafionych (B).
        
    Zwraca:
        RMSE (float): obliczony błąd RMS.
        percentage_bad (float): procent pikseli, dla których |różnica| > bad_threshold.
    """
    # Wczytanie obrazów
    img1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        raise ValueError("Failed to load file.")


    # Sprawdzenie, czy obrazy mają ten sam rozmiar
    if img1.shape != img2.shape:
        raise ValueError("Different sizes of images.")

    # Konwersja do float32 dla precyzyjnych obliczeń
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Obliczenie RMS: sqrt(średnia z (różnica^2))
    MSE = np.square(np.subtract(img1,img2)).mean() 
    RMSE = math.sqrt(MSE)
    
    # Obliczenie procentu źle dopasowanych pikseli,|różnica|>bad_threshold
    bad_pixels = np.sum(cv2.absdiff(img1, img2) > bad_threshold)
    total_pixels = cv2.absdiff(img1, img2).size
    percentage_bad = (bad_pixels / total_pixels) * 100.0

    return RMSE, percentage_bad
# Ustalanie ścieżek do plików
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comparing computed depth map and ground truth.'
    )
    parser.add_argument('image1', help='Path to calculated depth map.png')
    parser.add_argument('image2', help='Path to ground truth map.png')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Threshold for bad matching pixels (default 1.0)')
    args = parser.parse_args()

    RMSE, percentage_bad = quality(args.image1, args.image2, args.threshold)
    print(f"RMS error: {RMSE:.2f}")
    print(f"Percentage of bad matching pixels (B): {percentage_bad:.2f}%")
