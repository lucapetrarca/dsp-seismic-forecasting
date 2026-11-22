#Corre todos los notebooks del proyecto en orden y guarda los resultados en cada uno.
#Se necesita tener los chunks en data/raw (ver scripts/download_stead.py). Tarda alrededor de una hora.
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = Path(__file__).resolve().parent / 'notebooks'
ORDEN = [
    'sanity_check.ipynb',
    'carga_y_split.ipynb',
    'eda_seismic_noise.ipynb',
    'filters_design_freq.ipynb',
    'deteccion_p.ipynb',
    'prediccion_s.ipynb',
    'evaluacion_final.ipynb',
]

for nombre in ORDEN:
    inicio = time.time()
    print(f'Corriendo {nombre}...', flush=True)
    nb = nbformat.read(NOTEBOOKS / nombre, as_version=4)
    #Se corre desde la carpeta notebooks, porque los notebooks importan src con sys.path.append('..')
    NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': str(NOTEBOOKS)}}).execute()
    nbformat.write(nb, NOTEBOOKS / nombre)
    print(f'  listo en {(time.time() - inicio) / 60:.1f} min', flush=True)

print('Terminado. Las figuras quedan en docs/figures y la CNN en models/cnn.pt')
