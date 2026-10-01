import os
import glob
import re
import argparse
from PIL import Image

# Configurazione del parser degli argomenti da riga di comando
parser = argparse.ArgumentParser(description="Compilatore GIF animate da sequenze di immagini.")
parser.add_argument("-seg", type=int, default=70, help="Dimensione del segmento (es. 70)")
parser.add_argument("-method", type=str, default="hybrid", help="Metodo utilizzato (es. hybrid, max_norm)")
parser.add_argument("-energy", type=str, default="E2-4keV", help="Range di energia (es. E2-4keV, E4-8keV)")

# Parsing degli argomenti
args = parser.parse_args()

segment_size = args.seg
method = args.method
energy_range = args.energy

# Cartella principale aggiornata con la nuova struttura (es. seg80s_E4-8keV/images)
images_dir = r"C:\Users\gabri\OneDrive\Fisica\TESI\QPO_python\QPO_swift\Obs1\ALLDU_v"
subfolder_name = f"seg{segment_size}s_{energy_range}"
images_dir = os.path.join(images_dir, subfolder_name, "images")

print("\n" + "="*50)
print(f"🎬 COMPILAZIONE GIF: {energy_range} | {method.upper()} | SEG {segment_size}s")
print("="*50)

# Nuovo prefisso del file (rimosso energy_range poiché ora è nella cartella)
file_prefix = f"polar_{method}_{segment_size}_"
search_pattern = os.path.join(images_dir, f"{file_prefix}*.png")
all_files = glob.glob(search_pattern)

# Funzione di ordinamento basata sul valore numerico della soglia estratto dal nome del file
def extract_threshold(file_path):
    filename = os.path.basename(file_path)
    match = re.search(rf"{re.escape(file_prefix)}([\d.]+)\.png", filename)
    if match:
        return float(match.group(1))
    return 0.0

# Ordina i file in base al valore numerico della soglia
frames_paths = sorted(all_files, key=extract_threshold)

if frames_paths:
    gif_name = f"polar_contour_evolution_{energy_range}_{segment_size}s_{method}.gif"
    gif_output_path = os.path.join(images_dir, gif_name)
    
    print(f"[INFO] Trovati {len(frames_paths)} frame. Ordinamento e compilazione GIF in corso...")
    
    # Carica tutte le immagini in una lista
    images_list = [Image.open(f) for f in frames_paths]
    
    base_image = images_list[0]
    append_images_tuple = images_list[1:]
    
    # Salva la GIF animata
    base_image.save(
        gif_output_path,
        format='GIF',
        save_all=True,
        append_images=append_images_tuple,
        duration=500,  # 500 ms per frame (0.5 secondi)
        loop=0         # Loop infinito
    )
    
    # Chiude i file per liberare la memoria
    for img in images_list:
        img.close()
        
    print(f"🎉 GIF CREATA CON SUCCESSO!\n--> {gif_output_path}\n")
else:
    print(f"❌ Nessuna immagine corrispondente a '{file_prefix}*.png' trovata nella cartella:\n--> {images_dir}")