#!/usr/bin/env python3
import os
import subprocess

def main():
    # Define aquí la ruta de la carpeta padre que contiene las subcarpetas a procesar
    parent_path = "/home/admin_flw/TUD_Thesis/flw_dataset/zivid"
    
    # Verificar que el path exista
    if not os.path.isdir(parent_path):
        print(f"Error: El path '{parent_path}' no existe o no es un directorio.")
        return
    
    # Iterar sobre cada elemento en la carpeta padre
    for item in os.listdir(parent_path):
        # Construir la ruta completa de la subcarpeta
        X_PATH = os.path.join(parent_path, item)
        
        # Verificar que sea un directorio
        if os.path.isdir(X_PATH):
            # Construir las rutas de las subcarpetas requeridas
            images_dir = os.path.join(X_PATH, "segmentation")
            rosbag_dir = os.path.join(X_PATH, "rosbag")
            rgb_dir    = os.path.join(X_PATH, "rgb")
            
            # Comprobar si las subcarpetas requeridas existen
            if all(os.path.isdir(d) for d in [images_dir, rosbag_dir, rgb_dir]):
                print(f"Procesando carpeta: {X_PATH}")
                
                # Ejecutar el primer script con X_PATH como argumento
                try:
                    subprocess.run([
                        "python3",
                        "/home/admin_flw/TUD_Thesis/RVT_FLW_Dataset_Github-main/FLW_Dataset/update/events_to_grayscaleV2.py",
                        X_PATH
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error al ejecutar events_to_grayscaleV2.py en {X_PATH}: {e}")
                    continue
                
                # Ejecutar el segundo script con X_PATH como argumento
                try:
                    subprocess.run([
                        "python3",
                        "/home/admin_flw/TUD_Thesis/RVT_FLW_Dataset_Github-main/FLW_Dataset/update/create_labelsv5.py",
                        X_PATH
                    ], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error al ejecutar create_labelsv5.py en {X_PATH}: {e}")
                    continue
            else:
                # Si falta alguna subcarpeta, se salta este directorio
                print(f"Skipping this folder (faltan subcarpetas) -> {X_PATH}")
        else:
            # Si no es un directorio, también se omite
            print(f"Skipping this item (no es un directorio) -> {X_PATH}")

if __name__ == "__main__":
    main()
