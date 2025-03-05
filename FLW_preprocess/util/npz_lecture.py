
import numpy as np

def load_npz_data(file_path, output_txt_path):
    # Cargar el archivo .npz
    data = np.load(file_path)
    
    # Leer las matrices
    labels = data['labels']
    objframe_idx_2_label_idx = data['objframe_idx_2_label_idx']
    
    # Escribir las matrices en un archivo .txt
    with open(output_txt_path, 'w') as f:
        f.write("labels:\n")
        np.savetxt(f, labels, fmt='%s')
        f.write("\nobjframe_idx_2_label_idx:\n")
        np.savetxt(f, objframe_idx_2_label_idx, fmt='%s')

# Ejemplo de uso
file_path = '/home/admin_flw/TUD_Thesis/flw_dataset/zivid_SSOD/zivid_2/images/visualized_images/labels_v2/labels.npz'  # Reemplaza con la ruta de tu archivo
output_txt_path = 'LABELS_z1.txt'  # Reemplaza con la ruta del archivo de salida
load_npz_data(file_path, output_txt_path)

print("Matrices guardadas en", output_txt_path)
