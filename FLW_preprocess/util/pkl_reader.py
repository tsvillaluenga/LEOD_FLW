import pickle
import json

def read_pkl_and_save_txt(pkl_file, txt_file):
    try:
        # Cargar el archivo .pkl
        with open(pkl_file, 'rb') as file:
            data = pickle.load(file)
        
        # Convertir los datos a un formato legible (JSON para mayor claridad)
        with open(txt_file, 'w', encoding='utf-8') as output_file:
            json.dump(data, output_file, indent=4, ensure_ascii=False)
        
        print(f"Datos guardados en {txt_file}")
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    pkl_path = "/media/admin_flw/Volume/LEOD_FLW/data/genx_utils/splits/flw_dataset/ssod_0.010-off0.pkl"  # Cambia esto por la ruta de tu archivo .pkl
    txt_path = "datosPKL_pallets.txt"     # Nombre del archivo de salida
    read_pkl_and_save_txt(pkl_path, txt_path)
