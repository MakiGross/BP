import cv2
import numpy as np
import os
import csv

def evaluate_accuracy(target_mask, predicted_edge):
    # Převedeme na binární (0 nebo 1)
    # Práh 128 je standard, u tenkých hran z ArcGIS možno snížit na 30
    target = (target_mask > 128).astype(np.uint8)
    pred = (predicted_edge > 30).astype(np.uint8)

    # Výpočet pixelů
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# --- NASTAVENÍ ---
gt_path = "GT_vykres.jpg" # Váš ruční výkres
results_folder = "vsechny_vysledky" # Složka, kde máte Canny, Sobel, HED, ArcGIS...
output_csv = "finalni_srovnani.csv"

# Načtení Ground Truth
gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
if gt_img is None:
    print("Chyba: Maska ručního výkresu nenalezena!")
    exit()

stats_list = []

print("Zahajuji porovnávání s ručním výkresem...")

for filename in os.listdir(results_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        file_path = os.path.join(results_folder, filename)
        
        # Načtení výsledku z jakéhokoliv zdroje
        res_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if res_img is None: continue
        
        # Kontrola rozměrů (musí být stejné pro porovnání pixelů)
        if res_img.shape != gt_img.shape:
            res_img = cv2.resize(res_img, (gt_img.shape[1], gt_img.shape[0]))

        # Výpočet metrik
        prec, rec, f1 = evaluate_accuracy(gt_img, res_img)
        
        # Určení zdroje (odhadneme z názvu souboru)
        source = "Neznámý"
        if "canny" in filename.lower(): source = "Canny (OpenCV)"
        elif "sobel" in filename.lower(): source = "Sobel (OpenCV)"
        elif "hed" in filename.lower() and "arcgis" not in filename.lower(): source = "HED (OpenCV)"
        elif "arcgis" in filename.lower(): source = "HED (ArcGIS)"

        stats_list.append([source, filename, round(prec, 4), round(rec, 4), round(f1, 4)])
        print(f"Hotovo: {filename} -> F1: {round(f1, 3)}")

# Uložení do CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Metoda", "Soubor", "Precision", "Recall", "F1_Score"])
    writer.writerows(stats_list)

print(f"\nTabulka byla vytvořena: {output_csv}")