import cv2
import numpy as np
import os
import csv
from pathlib import Path

# --- KONFIGURACE ---
output_dir = "output_edges"
low_t = 100
high_t = 200
valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')

os.makedirs(output_dir, exist_ok=True)
stats_data = []

def get_stats(img):
    total_px = img.size
    edge_px = np.count_nonzero(img > 30)
    density = (edge_px / total_px) * 100
    avg_int = np.mean(img[img > 30]) if edge_px > 0 else 0
    return edge_px, density, avg_int

print("Spouštím Sobel a Canny detekci...")

current_dir = os.getcwd()
for root, dirs, files in os.walk(current_dir):
    if output_dir in root: continue
    
    for filename in files:
        if filename.lower().endswith(valid_extensions):
            full_path = os.path.join(root, filename)
            img = cv2.imread(full_path)
            if img is None: continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # --- SOBEL ---
            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(sobelx**2 + sobely**2)
            sobel_out = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            
            px_s, dens_s, int_s = get_stats(sobel_out)
            stats_data.append(["Sobel", filename, px_s, dens_s, int_s])
            cv2.imwrite(os.path.join(output_dir, f"{Path(filename).stem}_sobel.jpg"), sobel_out)
            
            # --- CANNY ---
            canny_out = cv2.Canny(gray, low_t, high_t)
            
            px_c, dens_c, int_c = get_stats(canny_out)
            stats_data.append(["Canny", filename, px_c, dens_c, int_c])
            cv2.imwrite(os.path.join(output_dir, f"{Path(filename).stem}_canny.jpg"), canny_out)
            
            print(f"Zpracováno: {filename}")

# Zápis do CSV
with open('statistiky_sobel_canny.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Algoritmus", "Soubor", "Pocet_Pixelu", "Hustota_Procenta", "Prumerna_Intenzita"])
    writer.writerows(stats_data)

print("Hotovo. Statistiky uloženy do statistiky_sobel_canny.csv")