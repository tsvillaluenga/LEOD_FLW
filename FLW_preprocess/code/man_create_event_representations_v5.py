import os
import re
import rosbag
import numpy as np
import h5py

def load_timestamps_set_from_npz(npz_path):
    """
    Verifica si timestamp_ns existe en la primera columna de npz_path.
    Retorna True si existe, False en caso contrario.
    """
    # Cargamos los datos (usa allow_pickle=True si tu .npz lo requiere)
    data = np.load(npz_path, allow_pickle=True)
    
    # Extraemos la matriz "labels" (ajusta el nombre según tu .npz)
    labels = data['labels']
    
    # labels.shape se espera (N,). Cada elemento de 'labels' 
    # debe ser una secuencia con 8 elementos, 
    # donde el [0] es el timestamp grande.
    first_col = [row[0] for row in labels]
    
    # Convertimos a int64, por si vienen como float
    first_col = np.array(first_col, dtype=np.int64)
    
    # Retornamos un set para búsquedas en O(1) promedio
    return set(first_col)


def process_bag_directory(output_base_dir):
    """
    Esta función contiene la lógica para procesar una carpeta (output_base_dir) que 
    contenga exactamente un archivo .bag. Realiza las mismas operaciones que tu 
    script original.
    """

    # =========================================================================
    # 1. LOCALIZAR EL ARCHIVO .BAG (se asume que sólo hay uno en la carpeta)
    # =========================================================================
    bag_files = [f for f in os.listdir(output_base_dir) if f.endswith('.bag')]
    if len(bag_files) == 0:
        print(f"[ADVERTENCIA] No se encontró ningún archivo '.bag' en {output_base_dir}. Se omite esta carpeta.")
        return
    elif len(bag_files) > 1:
        print(f"[ADVERTENCIA] Se encontró más de un archivo '.bag' en {output_base_dir}. Se tomará el primero y se omiten los demás.")
    
    rosbag_path = os.path.join(output_base_dir, bag_files[0])
    print(f"\nProcesando carpeta: {output_base_dir}")
    print(f"Archivo .bag encontrado: {rosbag_path}")

    # =========================================================================
    # 2. DEFINIR RUTAS DE SALIDA
    # =========================================================================
    # Repite la lógica original para definir tus rutas de salida
    output_dir = os.path.join(output_base_dir, "event_representations_v2/stacked_histogram_dt=50_nbins=10")
    hdf5_file_path = os.path.join(output_dir, 'event_representations.h5')
    timestamps_file_path = os.path.join(output_dir, 'timestamps_us.npy')
    objframe_idx_file_path = os.path.join(output_dir, 'objframe_idx_2_repr_idx.npy')
    external_timestamps_file_path = os.path.join(output_base_dir, "images/visualized_images/labels_v2/timestamps_us.npy")
    visualized_img_path = os.path.join(output_base_dir, "images/visualized_images")
    labels_path = os.path.join(output_base_dir, "images/visualized_images/labels_v2/labels.npz")
    
    # =========================================================================
    # 3. CREACIÓN DE DIRECTORIOS
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory creado en: {output_dir}")

    # =========================================================================
    # 4. INICIALIZACIÓN DE VARIABLES
    # =========================================================================
    histogram_bins = 20      # Número de bins en cada histograma
    time_interval = 50000000    # Intervalo de 50 us (en microsegundos)
    height, width = 240, 304 # TARGET DIMENSIONS
    
    # ORIGINAL DIMENSIONS
    orig_height, orig_width = 480, 640  # Original event image size

    # Scaling factors
    scale_x = width / orig_width
    scale_y = height / orig_height

    stacked_histograms = []
    timestamps = []
    current_histogram = np.zeros((histogram_bins, height, width), dtype=np.uint8)
    start_time = None
    relative_start_time = 0
    current_bin_start_time = 0

    print("Variables de histograma inicializadas.")

    # =========================================================================
    # 5. CARGA DE TIMESTAMPS EXTERNOS
    # =========================================================================
    try:
        external_timestamps = np.load(external_timestamps_file_path)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo de timestamps externos: {external_timestamps_file_path}")
        print("Omitiendo esta carpeta...\n")
        return

    print(f"Timestamps externos cargados desde: {external_timestamps_file_path}")

    # Ajuste: si los timestamps externos están en ns, convertirlos a µs
    external_timestamps = external_timestamps ########// 1000

    # =========================================================================
    # 6. FUNCIÓN PARA OBTENER TIEMPOS MÍNIMO Y MÁXIMO DE LAS IMÁGENES PNG
    # =========================================================================
    def low_high_num_finder(folder_path):
        # Verifica que la carpeta exista
        if not os.path.isdir(folder_path):
            raise ValueError(f"La carpeta '{folder_path}' no existe.")
        
        files_png = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        if not files_png:
            raise ValueError("No se encontraron archivos .png en la carpeta.")
        
        # Extrae números de los nombres de archivo
        numbers = []
        for archivo in files_png:
            match = re.search(r'\d+', archivo)  # Busca un número en el nombre
            if match:
                numbers.append(int(match.group()))
        
        if not numbers:
            raise ValueError("No se encontraron números en los nombres de los archivos .png.")
        
        # Encuentra el valor mínimo y máximo
        lower = min(numbers)
        higher = max(numbers)
        return lower, higher

    # =========================================================================
    # 7. DEFINICIÓN DE TIEMPOS DE INICIO Y FIN
    # =========================================================================
    try:
        specified_start_time, specified_end_time = low_high_num_finder(visualized_img_path)
    except ValueError as e:
        print(f"[ERROR]: {str(e)}")
        print("Omitiendo esta carpeta...\n")
        return

    # timestamps en ns
    specified_start_time = specified_start_time 
    specified_end_time = specified_end_time 
    print(f"Specified_start_time: {specified_start_time}")
    print(f"Specified_end_time: {specified_end_time}")

    # Ajuste del tiempo de inicio para que coincida con el intervalo
    start_time_adjustment = specified_start_time - (specified_start_time % time_interval)
    print(f"Tiempo de inicio ajustado a {start_time_adjustment} para incluir timestamps especificados")

    # =========================================================================
    # 8. PROCESAMIENTO DE EVENTOS DESDE EL ROSBAG
    # =========================================================================
    
    # Comprobacion timestamps labels y eventos
    timestamps_set = load_timestamps_set_from_npz(labels_path)
    print(f"Se han cargado {len(timestamps_set)} timestamps desde {labels_path}.")
    
    try:
        with rosbag.Bag(rosbag_path, 'r') as bag:
            print(f"Archivo ROS bag abierto: {rosbag_path}")
            try:
                for topic, msg, t in bag.read_messages(topics=['/dvxplorer_left/events']):
                    for event in msg.events:
                        # Convierte el timestamp del evento a microsegundos
                        timestamp_ns = t.to_nsec() 
                    ### üDEBUG] ### print(timestamp_ns)
                        
                        # Verifica si existe en la primera columna de tu archivo .npz
                        if timestamp_ns not in timestamps_set:
                            continue
                        else:
                            # # Inicializa start_time con el primer evento
                            # if start_time is None:
                            #     start_time = timestamp_ns
                            #     relative_start_time = start_time_adjustment
                            #     current_bin_start_time = start_time_adjustment
                            #     print(f"Tiempo de inicio configurado a {start_time} microsegundos")
                    ### üDEBUG] ### print(f"{timestamp_ns}: IN")
                            if start_time is None:
                                start_time = timestamp_ns
                                start_time_adjustment = start_time - (start_time % time_interval)  # Ajustado al primer evento válido
                                relative_start_time = start_time_adjustment
                                current_bin_start_time = start_time_adjustment
                                print(f"Tiempo de inicio ajustado a {start_time_adjustment} para incluir timestamps especificados")

                            
                            # Calcula el tiempo relativo a partir del tiempo de inicio ajustado
                            relative_time = timestamp_ns - start_time + start_time_adjustment
                            if relative_time < start_time_adjustment:
                    ### üDEBUG] ### print(f"{timestamp_ns}: relative_time < start_time_adjustment")
                                continue

                            # Determina el índice de bin correspondiente para el evento
                            bin_idx = (relative_time - current_bin_start_time) // time_interval
                            #print(f"{timestamp_ns}: ¿¿{bin_idx} >= {histogram_bins}??")
                            #print(f"{timestamp_ns}: ------------ ¿¿¿{relative_time} - ({current_bin_start_time}, {current_bin_start_time+time_interval}) = {relative_time - current_bin_start_time}???")
                            
                            
                            while relative_time >= (current_bin_start_time + time_interval): ##bin_idx >= histogram_bins:
                                stacked_histograms.append(current_histogram)
                                timestamps.append([current_bin_start_time, current_bin_start_time + time_interval])
                                print(f"Histograma y timestamps agregados. Total histograms: {len(stacked_histograms)}")
                                
                                # Reinicia el histograma para el siguiente bin
                                current_histogram = np.zeros((histogram_bins, height, width), dtype=np.uint8)
                                current_bin_start_time += time_interval #(time_interval*histogram_bins)
                                relative_start_time = current_bin_start_time
                    ### üDEBUG] ### print(f"REINICIO HIST: ------------ current_bin_start_time: {current_bin_start_time} .. = {relative_time - current_bin_start_time}")
                                bin_idx = (relative_time - current_bin_start_time) // time_interval
                                print(f"Nuevo bin de histograma inicializado. Tiempo relativo actualizado a {relative_start_time} microsegundos")

                            if current_bin_start_time > specified_end_time:
                                print(f"Se alcanzó el tiempo final: {specified_end_time}. Finalizando procesamiento.")
                                raise StopIteration
                            
                            # MAP EVENTS TO NEW RESOLUTION
                            new_x = int(event.x * scale_x)
                            new_y = int(event.y * scale_y)
                            
                            # Rellena el histograma con el evento
                            if 0 <= new_x < width and 0 <= new_y < height:
                    ### üDEBUG] ### print("Se GUARDA event en histograma")
                                current_histogram[int(bin_idx), new_y, new_x] += 1

            except (KeyboardInterrupt, StopIteration):
                print("Deteniendo procesamiento de datos...")

    except Exception as e:
        print(f"[ERROR] Ocurrió un problema al leer {rosbag_path}: {str(e)}")
        print("Omitiendo esta carpeta...\n")
        return

    # Asegurarse de cubrir hasta el tiempo especificado
    while current_bin_start_time <= specified_end_time:
        stacked_histograms.append(current_histogram)
        timestamps.append([current_bin_start_time, current_bin_start_time + time_interval])
        print(f"Histograma y timestamps agregados. Total histograms: {len(stacked_histograms)}")
        
        current_histogram = np.zeros((histogram_bins, height, width), dtype=np.uint8)
        current_bin_start_time += time_interval

    # =========================================================================
    # 9. GUARDAR TIMESTAMPS EN NPY
    # =========================================================================
    print(f"Guardando timestamps en: {timestamps_file_path}")
    np.save(timestamps_file_path, np.array(timestamps))
    print("Timestamps guardados.")

    # =========================================================================
    # 10. GUARDAR HISTOGRAMAS EN ARCHIVO HDF5
    # =========================================================================
    print(f"Guardando histogramas en: {hdf5_file_path}")
    with h5py.File(hdf5_file_path, 'w') as f:
        blosc_filter = 32001  # ID del filtro Blosc en HDF5
        chunk_shape = (1, histogram_bins, height, width)  # Definir chunk shape
        f.create_dataset(
            'data', 
            data=np.array(stacked_histograms),
            compression=blosc_filter,  # Usar BLOSC
            chunks=chunk_shape
        )
    print("Histogramas guardados en archivo HDF5 (BLOSC compression).")
    # # with h5py.File(hdf5_file_path, 'w') as f:
    # #     f.create_dataset('data', data=np.array(stacked_histograms))
    # # print("Histogramas guardados en archivo HDF5.")

    # =========================================================================
    # 11. GENERAR Y GUARDAR OBJFRAME_IDX_2_REPR_IDX BASADO EN LOS TIMESTAMPS EXTERNOS
    # =========================================================================
    print(f"Generando objframe_idx_2_repr_idx basados en los timestamps externos...")

    objframe_idx_2_repr_idx = []
    unmatched_timestamps = []
    num_histograms = len(timestamps)

    # for ts in external_timestamps:
    #     matched = False
    #     for idx, (ts_start, ts_end) in enumerate(timestamps):
    #         if ts_start <= ts <= ts_end:
    #             objframe_idx_2_repr_idx.append(idx)
    #             matched = True
    #             break
    #     if not matched:
    #         unmatched_timestamps.append(ts)
    #         objframe_idx_2_repr_idx.append(-1)  # Añadir -1 si no se encuentra un bin
    #         print(f"[WARNING] El timestamp externo {ts} no coincidió con ningún bin de histograma.")
    
    for ts in external_timestamps:
        matched = False
        for idx, (ts_start, ts_end) in enumerate(timestamps):
            if ts_start <= ts <= ts_end:
                objframe_idx_2_repr_idx.append(idx)
                matched = True
                break
        if not matched:
            unmatched_timestamps.append(ts)
            objframe_idx_2_repr_idx.append(len(timestamps) - 1)  # En lugar de -1, usa el último índice válido


    # Asegurar que ningún índice sea mayor al número de histogramas
    objframe_idx_2_repr_idx = np.array(objframe_idx_2_repr_idx)
    objframe_idx_2_repr_idx[objframe_idx_2_repr_idx >= num_histograms] = num_histograms - 1  # Corregir valores fuera de rango
    objframe_idx_2_repr_idx[objframe_idx_2_repr_idx < 0] = 0  # Límite mínimo para evitar errores

    # Ordenar el array en orden ascendente
    objframe_idx_2_repr_idx = sorted(objframe_idx_2_repr_idx)
    print(f"objframe_idx_2_repr_idx ordenado: {objframe_idx_2_repr_idx}")
    
    ##################################################################
    # Asegurar que objframe_idx_2_repr_idx es un numpy array
    objframe_idx_2_repr_idx = np.array(objframe_idx_2_repr_idx, dtype=int)

    # Verificar índices antes de guardar
    print(f"Cantidad total de histogramas generados: {len(timestamps)}")
    print(f"Cantidad total de objframe_idx_2_repr_idx generados: {len(objframe_idx_2_repr_idx)}")
    # Contar cuántos índices están fuera del rango válido
    out_of_range_count = sum(idx >= len(timestamps) for idx in objframe_idx_2_repr_idx)
    print(f"Índices fuera de rango: {out_of_range_count}")

    # Mostrar los primeros 10 índices fuera de rango
    out_of_range_examples = [idx for idx in objframe_idx_2_repr_idx if idx >= len(timestamps)]
    print(f"Ejemplo de índices fuera de rango: {out_of_range_examples[:10]}")

    # Forzar los valores dentro del rango válido antes de guardar
    objframe_idx_2_repr_idx = np.clip(objframe_idx_2_repr_idx, 0, len(timestamps) - 1)

    ####################################################################



    # Guardar el archivo corregido
    np.save(objframe_idx_file_path, objframe_idx_2_repr_idx)
    print(f"Índices de frames guardados basado en timestamps externos (total: {len(objframe_idx_2_repr_idx)}).")


    # Mostrar advertencias si hubo timestamps sin bin asignado
    if unmatched_timestamps:
        print(f"Advertencia: {len(unmatched_timestamps)} timestamps externos no coincidieron con ningún bin de histograma.")
    else:
        print("Todos los timestamps externos coincidieron con bins de histograma.")

    print("Creación del dataset completada.")
    print("------------------------------------------------------------\n")


if __name__ == "__main__":
    # =========================================================================
    # Directorio raíz que contiene múltiples subcarpetas, cada una con un .bag
    # =========================================================================
    main_directory = "/root/TUD_Thesis/flw_dataset/zivid"  # <--- AJUSTA ESTA RUTA A TU NECESIDAD

    # Iteramos sobre cada elemento del directorio principal
    for folder_name in os.listdir(main_directory):
        # Construimos la ruta completa de la subcarpeta
        subdir_path = os.path.join(main_directory, folder_name)

        # Solo nos interesa si es un directorio
        if os.path.isdir(subdir_path):
            # Llamamos a la función para procesar esa carpeta en particular
            process_bag_directory(subdir_path)

    print("Procesamiento completado para todas las subcarpetas.")
