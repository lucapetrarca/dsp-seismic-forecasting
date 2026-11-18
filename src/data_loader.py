#Carga de STEAD: metadata, trazas y split train/test con validación para luego tomar decisiones.
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / 'data' / 'raw'
PROCESSED = RAIZ / 'data' / 'processed'
CHUNKS = (1, 2, 6)

def leer_metadata(chunks=CHUNKS):
    #Junta los CSV de los chunks en una sola tabla, con una columna que dice de qué chunk es cada traza.
    tablas = []
    for chunk in chunks:
        df = pd.read_csv(RAW / f'chunk{chunk}.csv', low_memory=False, dtype={'source_id': str})
        df['chunk'] = chunk
        tablas.append(df)
    return pd.concat(tablas, ignore_index=True)

# Se utiliza el diccionario _hdf5 para no tener que reabrir los archivos
_hdf5 = {}


def cargar(trace_name, chunk):
    # Devuelve la traza como array (6000, 3), con columnas E, N, Z.
    if chunk not in _hdf5:
        _hdf5[chunk] = h5py.File(RAW / f'chunk{chunk}.hdf5', 'r')
    return np.array(_hdf5[chunk]['data/' + trace_name])

def armar_split(df, frac_val=0.1, frac_test=0.2, semilla=0, rehacer=False):
    # Split en train, validación y test (70/10/20) y lo guarda en data/processed/split.csv.
    # Si el archivo ya existe lo lee en vez de rearmarlo, el split es siempre el mismo.
    ruta = PROCESSED / 'split.csv'
    if ruta.exists() and not rehacer:
        return pd.read_csv(ruta)
    split = pd.Series('train', index=df.index)
    frac_val_resto = frac_val / (1 - frac_test)

    # Los sismos se separan por evento
    sismos = df[df['trace_category'] == 'earthquake_local']
    gss = GroupShuffleSplit(n_splits=1, test_size=frac_test, random_state=semilla)
    i_resto, i_test = next(gss.split(sismos, groups=sismos['source_id']))
    resto = sismos.iloc[i_resto]
    gss = GroupShuffleSplit(n_splits=1, test_size=frac_val_resto, random_state=semilla)
    _, i_val = next(gss.split(resto, groups=resto['source_id']))
    split.loc[sismos.index[i_test]] = 'test'
    split.loc[resto.index[i_val]] = 'val'

    # El Ruido se reparte al azar
    ruido = df[df['trace_category'] == 'noise']
    resto, test = train_test_split(ruido.index, test_size=frac_test, random_state=semilla)
    _, val = train_test_split(resto, test_size=frac_val_resto, random_state=semilla)
    split.loc[test] = 'test'
    split.loc[val] = 'val'

    tabla = pd.DataFrame({'trace_name': df['trace_name'], 'chunk': df['chunk'], 'split': split})
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return tabla