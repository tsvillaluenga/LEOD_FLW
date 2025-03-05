import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def transform_points(H, points):
    """
    Transforma una lista de puntos (x, y) con una matriz de transformación 3x3.
    Asume coordenadas homogéneas [x, y, 1].
    """
    transformed_points = []
    for x, y in points:
        point_hom = np.array([x, y, 1.0], dtype=np.float64)
        result = H @ point_hom  # Multiplicación de la matriz
        # Normalizar por la tercera componente
        if abs(result[2]) > 1e-12:
            x_t = result[0] / result[2]
            y_t = result[1] / result[2]
        else:
            # Evitar divisiones por cero en caso de valores anómalos
            x_t = result[0]
            y_t = result[1]
        transformed_points.append((x_t, y_t))
    return transformed_points


def actualizar_timestamps_json(json_labels_data, txt_file_path):
    """
        Matching RGB timestamps with Events timestamps of the labels
    """

    # 1. Construimos el diccionario leyendo el archivo TXT.
    mapa_timestamps = {}
    with open(txt_file_path, 'r', encoding='utf-8') as f_txt:
        for linea in f_txt:
            linea = linea.strip()
            if not linea:
                continue
            # Separamos por ' -- '
            partes = linea.split('--')
            if len(partes) == 2:
                tsp1, tsp2 = partes
                tsp1 = tsp1.strip()
                tsp2 = tsp2.strip()
                mapa_timestamps[tsp1] = tsp2

    # 2. Para cada fichero JSON en json_labels_data
    for json_file in json_labels_data:
        # Almacenamos las líneas modificadas temporalmente
        lineas_modificadas = []

        # Abrimos el JSON y lo leemos línea a línea
        with open(json_file, 'r', encoding='utf-8') as f_json:
            for linea_json in f_json:
                linea_json = linea_json.strip()
                if not linea_json:
                    continue
                # Convertimos cada línea en dict
                try:
                    data = json.loads(linea_json)
                except json.JSONDecodeError:
                    # Si una línea no es JSON válido, la ignoramos o la añadimos tal cual
                    lineas_modificadas.append(linea_json)
                    continue

                # Verificamos si el timestamp está en nuestro diccionario
                ts_actual = data.get("timestamp", "")
                if ts_actual in mapa_timestamps:
                    # Sustituimos por el TSP2
                    data["timestamp"] = mapa_timestamps[ts_actual]

                # Convertimos de nuevo a JSON para almacenar en lineas_modificadas
                lineas_modificadas.append(json.dumps(data))

        # Sobrescribimos el mismo fichero JSON con las líneas ya actualizadas
        with open(json_file, 'w', encoding='utf-8') as f_json_out:
            for linea_mod in lineas_modificadas:
                f_json_out.write(linea_mod + "\n")

def ordenar_json_por_timestamp(json_labels_data):
    """
    Ordena cada archivo contenido en json_labels_data (lista de rutas) 
    por el valor de 'timestamp'. Sobrescribe el archivo original con 
    las líneas ordenadas y retorna nuevamente la misma lista de rutas 
    (por si quieres volver a leer el contenido).
    """
    for file_path in json_labels_data:
        # Leemos todas las líneas (cada línea es un objeto JSON)
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
        
        # Parseamos cada línea como un objeto JSON
        data_list = [json.loads(line) for line in lines if line.strip()]

        # Ordenamos la lista según 'timestamp' (que convertimos a entero para un orden correcto)
        data_list.sort(key=lambda x: int(x["timestamp"]))

        # Escribimos de nuevo en el archivo todas las líneas en orden
        with open(file_path, 'w') as f:
            for item in data_list:
                f.write(json.dumps(item) + '\n')
    
    return json_labels_data # Devuelve los archivos ordenados

def main(MAIN_PATH):
    # Directorios basados en MAIN_PATH
    image_dir = Path(MAIN_PATH) / 'images'
    visualization_dir = image_dir / 'visualized_images'
    visualization_dir.mkdir(exist_ok=True)
    labels_dir = visualization_dir / 'labels_v2'
    labels_dir.mkdir(exist_ok=True)
    txt_file_path = Path(MAIN_PATH) / 'utils/event_images/mapping.txt'

    # Diccionarios y listas para almacenar resultados
    grouped_labels = {}
    labels = []
    image_timestamps = []
    objframe_idx_2_label_idx = []
    height, width = 240, 304

    # Archivos de etiquetado (buscamos en la carpeta "segmentation" dentro de MAIN_PATH) y ordenamos los datos
    json_labels_data = list((Path(MAIN_PATH) / 'segmentation').glob("rgb_bounding_box_labels_*.json"))
    json_labels_data = ordenar_json_por_timestamp(json_labels_data)
    

    # Matrices de transformación (3x3) para cada caso (NO se invierten).
    # Rectángulo 1 (Human)
    H_human = np.array([
        [0.1591,        -4.51e-15,     -11.4773],
        [1.47e-14,       0.1757,       -12.3579],
        [1.17e-16,      -7.02e-17,       1.0   ]
    ])

    # Rectángulo 2 (Otro elemento)
    H_other = np.array([
        [0.1647,        -1.23e-13,     -11.7647],
        [1.64e-13,       0.1619,        -3.0211],
        [1.81e-15,      -1.69e-15,       1.0   ]
    ])

    # Matchin timestamps RGB - Events
    actualizar_timestamps_json(json_labels_data, txt_file_path)
    
    
    print("Processing JSON files...")
    for json_file in json_labels_data:
        class_id = json_file.stem.split("_")[-1]  # Extrae "human" u otra etiqueta del nombre de archivo
        print(f"Processing JSON file: {json_file}")
        with open(json_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                timestamp = int(data["timestamp"])
                bbox = [
                    int(data["xmin"]),
                    int(data["ymin"]),
                    int(data["xmax"]),
                    int(data["ymax"]),
                    class_id
                ]
                if timestamp not in grouped_labels:
                    grouped_labels[timestamp] = []
                grouped_labels[timestamp].append(bbox)

    print(f"Grouped labels by timestamp: {list(grouped_labels.keys())[:10]}...")

    # Procesar cada timestamp e imagen
    print("Processing images...")
    for timestamp, bboxes in grouped_labels.items():
        # Buscar archivo de imagen con extensiones compatibles
        image_path = None
        for ext in ['.png', '.jpg']:
            temp_path = image_dir / f"{timestamp}{ext}"
            if temp_path.is_file():
                image_path = temp_path
                break
        
        if image_path is None:
            print(f"Image for timestamp {timestamp} not found in formats .png or .jpg")
            continue

        cv_image = cv2.imread(str(image_path))
        frame_labels = []

        print(f"Drawing bounding boxes for timestamp: {timestamp}")
        for bbox in bboxes:
            x1, y1, x2, y2, class_id = bbox
            points = [(x1, y1), (x2, y2)]
            
            # Escoger la matriz de transformación
            if class_id == 'human':
                H = H_human
                color = (0, 255, 0)
            else:
                H = H_other
                color = (255, 0, 0)
            
            # Transformar puntos
            transformed_points = transform_points(H, points)
            print(f"Puntos transformados para '{class_id}': {transformed_points}")
            
            # Asignar las nuevas coordenadas del bounding box
            x1_t = int(round(transformed_points[0][0])) - 2
            y1_t = int(round(transformed_points[0][1])) - 2
            x2_t = int(round(transformed_points[1][0])) + 2
            y2_t = int(round(transformed_points[1][1])) + 2
            
            # Evitar casos excepcionales
            if x1_t < 0: x1_t = 0
            elif x1_t > width: x1_t = width
            
            if x2_t < 0: x2_t = 0
            elif x2_t > width: x2_t = width
            
            if y1_t < 0: y1_t = 0
            elif y1_t > height: y1_t = height
            
            if y2_t < 0: y2_t = 0
            elif y2_t > height: y2_t = height

            
            if (x2_t - x1_t) <= 0 or (y2_t - y1_t) <= 0:
                continue
            
            # Dibuja el rectángulo y la etiqueta en la imagen
            cv2.rectangle(cv_image, (x1_t, y1_t), (x2_t, y2_t), color, 1)
            label = f"{class_id}"
            cv2.putText(cv_image, label, (x1_t, y1_t - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            # Guardar datos de la etiqueta (el ancho y alto se derivan de x2_t - x1_t, etc.)
            frame_labels.append((
                timestamp, 
                x1_t, 
                y1_t, 
                x2_t - x1_t,  # width
                y2_t - y1_t,  # height
                1 if class_id == 'human' else 0,  # ID de clase numérico
                1.0,  # confianza (dummy)
                1 if class_id == 'human' else 0     # track_id (dummy)
            ))

        if frame_labels:
            objframe_idx_2_label_idx.append(len(labels))  # índice del primer label de este frame
            labels.extend(frame_labels)
            image_timestamps.append(timestamp)

        # Guardar imagen visualizada
        output_path = visualization_dir / f"{timestamp}.png"
        cv2.imwrite(str(output_path), cv_image)
        print(f"Saved visualized image: {output_path}")

    # Convertir labels a array estructurado
    labels_dtype = [
        ('t', '<u8'), 
        ('x', '<f4'), 
        ('y', '<f4'), 
        ('w', '<f4'), 
        ('h', '<f4'), 
        ('class_id', 'u1'), 
        ('class_confidence', '<f4'), 
        ('track_id', '<u4')
    ]
    labels_array = np.array(labels, dtype=labels_dtype)

    # Convertir objframe_idx_2_label_idx a numpy array y guardarlo
    objframe_idx_2_label_idx = np.array(objframe_idx_2_label_idx, dtype=np.int64)
    np.savez(labels_dir / 'labels.npz',
             labels=labels_array,
             objframe_idx_2_label_idx=objframe_idx_2_label_idx)

    # Guardar timestamps
    image_timestamps = np.array(image_timestamps, dtype=np.int64)
    np.save(labels_dir / 'timestamps_us.npy', image_timestamps)

    print("Object detection and annotation complete. Visualized images saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script para procesar y anotar bounding boxes en imágenes."
    )
    parser.add_argument(
        "MAIN_PATH",
        type=str,
        help="Ruta principal al dataset (e.g., /root/TUD_Thesis/flw_dataset/"
    )
    args = parser.parse_args()
    main(args.MAIN_PATH)
