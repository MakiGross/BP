import cv2
import numpy as np
import os
import csv

# --- KONFIGURACE ---
protoPath = "deploy.prototxt"
modelPath = "hed_pretrained_bsds.caffemodel"
valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
output_dir = "output_hed"

# Načtení modelu
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
os.makedirs(output_dir, exist_ok=True)

stats_data = []

def get_stats(img):
    total_px = img.size
    edge_px = np.count_nonzero(img > 30)
    density = (edge_px / total_px) * 100
    avg_int = np.mean(img[img > 30]) if edge_px > 0 else 0
    return edge_px, density, avg_int

print("Spouštím HED detekci...")

current_dir = os.getcwd()
for root, dirs, files in os.walk(current_dir):
    if output_dir in root: continue
    
    for filename in files:
        if filename.lower().endswith(valid_extensions):
            full_path = os.path.join(root, filename)
            img = cv2.imread(full_path)
            if img is None: continue
            
            (H, W) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                                         mean=(105, 117, 123), swapRB=False, crop=False)
            net.setInput(blob)
            hed = net.forward()[0, 0, :, :]
            hed = (255 * hed).astype("uint8")
            
            # Statistiky
            px, dens, intensity = get_stats(hed)
            stats_data.append(["HED", filename, px, dens, intensity])
            
            # Uložení
            out_name = os.path.join(output_dir, f"HED_{filename.split('.')[0]}.png")
            cv2.imwrite(out_name, hed)
            print(f"Zpracováno: {filename}")

# Zápis do CSV
with open('statistiky_hed.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Algoritmus", "Soubor", "Pocet_Pixelu", "Hustota_Procenta", "Prumerna_Intenzita"])
    writer.writerows(stats_data)

print("Hotovo. Statistiky uloženy do statistiky_hed.csv")