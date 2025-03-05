import h5py
import numpy as np
import matplotlib.pyplot as plt
import hdf5plugin  # Importar el plugin necesario

# Ruta del archivo .h5
ruta_archivo = '/media/admin_flw/Volume/LEOD_FLW/datasets/flw_dataset/train/zivid_5/event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5'
ruta_salida_txt = "datos_extraidos.txt"

# Abrir el archivo y explorar su contenido
with h5py.File(ruta_archivo, 'r') as archivo:
    # Imprimir la estructura del archivo
    print("Estructura del archivo:")
    for clave in archivo.keys():
        print(f"- {clave}")
    
    # Acceder al dataset 'data'
    dataset = archivo['data']
    
    # Información general del dataset
    print("\nInformación del dataset 'data':")
    print(f"- Forma: {dataset.shape}")
    print(f"- Tipo de datos: {dataset.dtype}")
    print(f"- Atributos: {list(dataset.attrs.keys())}")
    
    # Leer los datos como un arreglo NumPy
    datos = dataset[:]
    print("\nEjemplo de datos (primeros 5 elementos si es 1D):")
    print(datos[:5] if datos.ndim == 1 else datos[:5, :5])  # Adaptado a 2D si es matriz

    # Guardar todos los datos en un archivo .txt
    print("\nGuardando los datos en un archivo de texto...")
    with open(ruta_salida_txt, "w") as f:
        f.write(f"# Datos extraídos del archivo: {ruta_archivo}\n")
        f.write(f"# Forma del dataset: {datos.shape}\n")
        f.write(f"# Tipo de datos: {datos.dtype}\n\n")
        
        # Guardado con formato adaptado a la dimensión
        if datos.ndim == 1:
            np.savetxt(f, datos, fmt="%.6f")  # Un solo vector
        elif datos.ndim == 2:
            np.savetxt(f, datos, fmt="%.6f", delimiter=", ")  # Matriz 2D
        else:
            # Para matrices 3D o superiores, guardamos cada "corte" en bloques separados
            for i in range(datos.shape[0]):  # Iterar sobre la primera dimensión
                f.write(f"\n# Slice {i}:\n")
                for j in range(datos.shape[1]):  # Iterar sobre la segunda dimensión
                    fila = ", ".join(map(str, datos[i, j]))  # Convertir la fila en texto
                    f.write(fila + "\n")  # Escribir fila en el archivo
                f.write("\n")  # Espaciado entre bloques

    print(f"Datos guardados en: {ruta_salida_txt}")

# Estadísticas básicas si los datos son numéricos
if np.issubdtype(datos.dtype, np.number):
    print("\nEstadísticas básicas del dataset 'data':")
    print(f"- Mínimo: {np.min(datos)}")
    print(f"- Máximo: {np.max(datos)}")
    print(f"- Media: {np.mean(datos)}")
    print(f"- Desviación estándar: {np.std(datos)}")

# Visualización si es 2D o 3D (como imágenes)
if datos.ndim == 2 or (datos.ndim == 3 and datos.shape[-1] in [1, 3]):
    print("\nVisualizando el dataset 'data':")
    plt.imshow(datos if datos.ndim == 2 else datos[:, :, 0], cmap='gray')
    plt.colorbar()
    plt.title('Visualización del dataset "data"')
    plt.show()
else:
    print("\nEl dataset no es compatible con visualización directa.")

