import os
import shutil

# Rutas de ejemplo (ajusta según tu escenario)
MAIN_DIR = "/home/admin_flw/TUD_Thesis/flw_dataset/zivid_SSOD"      # Directorio original
OUT_PATH = "/media/admin_flw/Volume/LEOD_FLW/datasets/flw_dataset"  # Directorio de destino

# Lista de las 2 subcarpetas (Y_DIRS) que deseas copiar en cada carpeta XX_PATH
subfolders_to_copy = ["images", "event_representations_v2"]

def replicate_structure(main_dir, out_dir, subfolders):
    """
    Crea en out_dir las mismas carpetas (XXO_PATH) que haya en main_dir (XX_PATH),
    y copia solo las subcarpetas listadas en 'subfolders' a cada una de ellas.
    Luego, mueve la subsubcarpeta 'images/visualized_images/labels_v2' a la raíz de xxo_path.
    """
    # 1. Crear el directorio base de salida si no existe
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 2. Recorrer las carpetas XX_PATH en MAIN_DIR
    for xx_folder in os.listdir(main_dir):
        xx_path = os.path.join(main_dir, xx_folder)
        
        # Verifica que sea un directorio (omite archivos)
        if os.path.isdir(xx_path):
            # Crea la carpeta correspondiente en OUT_PATH con el mismo nombre
            xxo_path = os.path.join(out_dir, xx_folder)
            os.makedirs(xxo_path, exist_ok=True)

            # 3. Copiar solo las subcarpetas especificadas en 'subfolders_to_copy'
            for subf in subfolders:
                source_subfolder = os.path.join(xx_path, subf)
                target_subfolder = os.path.join(xxo_path, subf)
                
                # Si existe la subcarpeta en la carpeta original, se copia
                if os.path.isdir(source_subfolder):
                    # Eliminamos destino si ya existe, para evitar errores de copytree
                    if os.path.exists(target_subfolder):
                        shutil.rmtree(target_subfolder)
                    
                    shutil.copytree(source_subfolder, target_subfolder)
                    print(f"Copiado: {source_subfolder} -> {target_subfolder}")
                else:
                    print(f"La subcarpeta '{subf}' no existe en '{xx_path}'. Se omite.")

            # 4. Mover la subcarpeta 'images/visualized_images/labels_v2' a xxo_path
            labels_v2_source = os.path.join(xxo_path, "images", "visualized_images", "labels_v2")
            labels_v2_target = os.path.join(xxo_path, "labels_v2")

            if os.path.exists(labels_v2_source):
                # Mover la carpeta
                shutil.move(labels_v2_source, labels_v2_target)
                print(f"Movido: {labels_v2_source} -> {labels_v2_target}")

                # Opcional: limpiar la ruta 'images/visualized_images' si quedó vacía
                visualized_images_dir = os.path.join(xxo_path, "images", "visualized_images")
                try:
                    # os.removedirs elimina la carpeta indicada y, recursivamente, 
                    # sus padres vacíos
                    os.removedirs(visualized_images_dir)
                except OSError:
                    # Ocurre si la carpeta no está vacía o no existe
                    pass

if __name__ == "__main__":
    replicate_structure(MAIN_DIR, OUT_PATH, subfolders_to_copy)
