import numpy as np

def read_npy_and_write_txt(npy_filename, txt_filename):
    try:
        # Cargar el archivo .npy
        data = np.load(npy_filename)

        # Verificar el tipo de datos y la forma
        print(f"Datos cargados desde {npy_filename}: {data.shape}, dtype={data.dtype}")

        # Guardar en un archivo de texto en formato legible
        np.savetxt(txt_filename, data, fmt='%d', delimiter=',', header='Timestamps (us)')

        print(f"Timestamps guardados en {txt_filename}")

    except FileNotFoundError:
        print(f"El archivo {npy_filename} no se encontró.")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")


read_npy_and_write_txt("/home/admin_flw/TUD_Thesis/flw_dataset/zivid/zivid_5/event_representations_v2/stacked_histogram_dt=50_nbins=10/objframe_idx_2_repr_idx.npy", "OB_evrep_z5.txt")
