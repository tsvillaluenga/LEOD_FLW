import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import argparse

def find_closest_timestamp(target_timestamp, timestamps):
    """Encuentra el timestamp más cercano en una lista a un timestamp objetivo."""
    return min(timestamps, key=lambda x: abs(x - target_timestamp))

def pad_timestamp(filename):
    """Asegura que un nombre de archivo tenga un timestamp de 19 dígitos, rellenando con ceros al final si es necesario."""
    stem = filename.stem
    if len(stem) < 19:
        padded_stem = stem.ljust(19, '0')
        return filename.with_name(f"{padded_stem}{filename.suffix}")
    return filename

def find_existing_file(timestamp, base_path):
    """Busca un archivo con un timestamp dado, removiendo ceros finales hasta encontrarlo."""
    while timestamp.endswith("0"):
        file_png = base_path / f"{timestamp}.png"
        file_jpg = base_path / f"{timestamp}.jpg"
        if file_png.exists():
            return file_png
        if file_jpg.exists():
            return file_jpg
        timestamp = timestamp[:-1]  # Remover un cero del final

    # Última búsqueda sin ceros finales
    file_png = base_path / f"{timestamp}.png"
    file_jpg = base_path / f"{timestamp}.jpg"
    if file_png.exists():
        return file_png
    if file_jpg.exists():
        return file_jpg
    return None

def create_grayscale_image(image_path, output_path):
    """Convierte una imagen a escala de grises y la guarda."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error al cargar la imagen: {image_path}")
        return

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(output_path), grayscale_image)

def main(MAIN_PATH):
    # Rutas dentro del dataset
    RGB_PATH = os.path.join(MAIN_PATH, "rgb")
    XX_PATH = os.path.join(MAIN_PATH, "rosbag")
    EVENT_IMAGES_PATH = os.path.join(MAIN_PATH, "utils", "event_images")
    OUTPUT_IMAGES_PATH = os.path.join(MAIN_PATH, "images")

    # Crear carpetas de salida si no existen
    os.makedirs(EVENT_IMAGES_PATH, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_PATH, exist_ok=True)

    # Obtener los nombres de los archivos en ambas carpetas
    rgb_images = sorted(Path(RGB_PATH).glob("*.png")) + sorted(Path(RGB_PATH).glob("*.jpg"))
    xx_images = sorted(Path(XX_PATH).glob("*.png")) + sorted(Path(XX_PATH).glob("*.jpg"))

    # Asegurar que los nombres tengan 19 dígitos
    rgb_images = [pad_timestamp(img) for img in rgb_images]
    xx_images = [pad_timestamp(img) for img in xx_images]

    rgb_images = sorted(rgb_images, key=lambda p: int(p.stem))
    xx_images = sorted(xx_images, key=lambda p: int(p.stem))

    print(f"Número de imágenes en /rgb: {len(rgb_images)}")
    print(f"Número de imágenes en XX_PATH: {len(xx_images)}")

    if not rgb_images:
        print("No se encontraron imágenes en la carpeta /rgb.")
    if not xx_images:
        print("No se encontraron imágenes en la carpeta XX_PATH.")

    if rgb_images and xx_images:
        print(f"Se encontraron {len(rgb_images)} imágenes en /rgb y {len(xx_images)} en XX_PATH.")

    # Convertir nombres de archivos a timestamps
    rgb_timestamps = [int(img.stem) for img in rgb_images]
    xx_timestamps = [int(img.stem) for img in xx_images]

    mapping_file_path = os.path.join(EVENT_IMAGES_PATH, "mapping.txt")

    with open(mapping_file_path, "w") as mapping_file:
        for rgb_img, rgb_timestamp in zip(rgb_images, rgb_timestamps):
            # 1) Evaluar el timestamp actual de /rgb
            print(f"Evaluando timestamp de /rgb: {rgb_timestamp}")

            # 2) Encontrar el timestamp más cercano en XX_PATH
            closest_timestamp = find_closest_timestamp(rgb_timestamp, xx_timestamps)
            print(f"Timestamp más cercano encontrado: {closest_timestamp}")

            # Buscar la imagen correspondiente en XX_PATH, ajustando ceros finales si es necesario
            closest_image = find_existing_file(str(closest_timestamp), Path(XX_PATH))

            if closest_image is None:
                print(f"Imagen no encontrada para timestamp: {closest_timestamp}")
                continue

            # Copiar la imagen a la carpeta event_images con el nombre del EVENT más cercano al RGB correspondiente
            original_stem = closest_image.stem  # Obtener el nombre de archivo sin extensión
            new_image_name = f"{original_stem}.png"
            new_image_path = Path(EVENT_IMAGES_PATH) / new_image_name

            print(f"Copiando imagen {closest_image} a {new_image_path}")
            shutil.copy(closest_image, new_image_path)

            # 3) Añadir información al archivo .txt
            mapping_file.write(f"{rgb_timestamp} -- {closest_image.stem}\n")
            print(f"Añadido al archivo mapping.txt: {rgb_timestamp} -- {closest_image.stem}")

            # 4) Procesar la imagen a escala de grises
            output_image_path = Path(OUTPUT_IMAGES_PATH) / new_image_name
            print(f"Convirtiendo imagen a escala de grises: {new_image_path} -> {output_image_path}")
            create_grayscale_image(new_image_path, output_image_path)

    print("Procesamiento completado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para procesar imágenes de un dataset.")
    parser.add_argument(
        "MAIN_PATH",
        type=str,
        help="Ruta principal al dataset (p. ej. /root/TUD_Thesis/flw_dataset/element_X)"
    )

    args = parser.parse_args()
    main(args.MAIN_PATH)
