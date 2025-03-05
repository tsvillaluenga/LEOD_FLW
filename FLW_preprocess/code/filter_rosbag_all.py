#!/usr/bin/env python3

import os
import rosbag
import shutil

# Ruta de la carpeta que contiene las rosbags a procesar (modifica esto con tu ruta específica)
rosbag_directory = "/root/bags"

# Define los topics que queremos conservar
topics_to_keep = {"/dvxplorer_left/events"}  # , "/rgb/image_raw"}

def filter_rosbag(input_bag_path, topics_to_keep):
    # Define el nombre de la nueva rosbag temporal
    temp_bag_path = input_bag_path + ".filtered"

    # Abre la rosbag original en modo lectura y crea una nueva rosbag para escribir
    with rosbag.Bag(input_bag_path, 'r') as in_bag, rosbag.Bag(temp_bag_path, 'w') as out_bag:
        for topic, msg, t in in_bag.read_messages():
            if topic in topics_to_keep:
                out_bag.write(topic, msg, t)

    # Reemplaza la rosbag original con la filtrada
    os.remove(input_bag_path)  # Elimina la original
    shutil.move(temp_bag_path, input_bag_path)  # Renombra la filtrada con el nombre original
    print(f"Rosbag filtrada creada: {input_bag_path}")

# Itera sobre todos los archivos .bag en la carpeta especificada
for filename in os.listdir(rosbag_directory):
    if filename.endswith(".bag"):
        rosbag_path = os.path.join(rosbag_directory, filename)
        print(f"Procesando: {rosbag_path}")
        filter_rosbag(rosbag_path, topics_to_keep)

