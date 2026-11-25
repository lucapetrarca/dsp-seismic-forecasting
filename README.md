# Seismic Early Warning System using Advanced DSP and Deep Learning

## Descripción del Proyecto
Este proyecto desarrolla un pipeline de procesamiento de señales para la alerta temprana de sismos, utilizando el Stanford Earthquake Dataset (STEAD). A partir de cada registro sísmico, el sistema detecta la llegada de la onda P mediante un detector STA/LTA sobre la señal filtrada, y con los 3 segundos que le siguen predice la amplitud pico de la onda S, comparando una regresión lineal con features clásicas y una red neuronal convolucional (CNN 1D).

El pipeline se divide en las siguientes etapas:
1. Carga de los datos y partición en train, validación y test por evento sísmico.
2. Análisis exploratorio: estacionariedad del ruido y comparación del espectro del ruido con el de los sismos.
3. Diseño de un filtro pasabanda Butterworth causal de 3 a 20 Hz, con la banda elegida a partir del análisis anterior.
4. Detección de la onda P con STA/LTA, con el umbral elegido sobre validación.
5. Predicción de la amplitud de la onda S con una regresión lineal y una CNN.
6. Evaluación final sobre test.

Todas las decisiones se toman sobre validación, y el conjunto de test se utiliza una única vez, en la evaluación final.

## Cómo reproducirlo

### 1. Entorno
Se utilizó Python 3.14. Las versiones exactas de las librerías están en `requirements.txt`.

```
git clone https://github.com/lucapetrarca/dsp-seismic-forecasting.git
cd dsp-seismic-forecasting
python -m venv .venv
.venv\Scripts\activate          (en Linux o Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

### 2. Datos
Los datos corresponden al dataset STEAD (Mousavi et al., 2019), disponible en https://github.com/smousavi05/STEAD bajo licencia CC BY 4.0. Cada traza es el registro de una estación durante 60 s a 100 Hz en tres canales (E, N y Z). El dataset se distribuye en 6 partes (chunks): el chunk 1 contiene únicamente ruido y los chunks 2 a 6 contienen sismos locales. En este proyecto se utilizan los chunks 1, 2 y 6.

Las formas de onda no están en el repositorio por su tamaño. Para descargarlas:

```
python scripts/download_stead.py
```

El script baja los chunks 1, 2 y 6 desde los mismos links de Google Drive que figuran en el repositorio de STEAD, los extrae en `data/raw/` y borra los .zip. Si la descarga se corta, alcanza con volver a correrlo. También se puede bajar un solo chunk, por ejemplo `python scripts/download_stead.py 6`.

Se recomienda usar el script en lugar de descomprimir a mano, ya que los .zip de STEAD superan los 4 GB y tienen mal escrito el índice ZIP64, por lo que `unzip` y el módulo `zipfile` de Python fallan. El script extrae cada archivo en orden y verifica su CRC.

Se necesitan alrededor de 70 GB libres durante la descarga y 50 GB al terminar. Al finalizar, `data/raw/` debe contener:

```
data/raw/chunk1.csv   data/raw/chunk1.hdf5
data/raw/chunk2.csv   data/raw/chunk2.hdf5
data/raw/chunk6.csv   data/raw/chunk6.hdf5
```

Los tamaños y la cantidad de trazas de cada chunk están en `metadata/MANIFEST.md`, y el notebook `sanity_check.ipynb` los verifica. La metadata completa de los tres chunks está también en `metadata/chunkN.csv.gz`.

### 3. Ejecución
Los notebooks de `notebooks/` se corren en este orden, ya que cada uno utiliza lo que genera el anterior:

1. `sanity_check.ipynb`: verificación de los datos.
2. `carga_y_split.ipynb`: partición en train, validación y test (se guarda en `data/processed/split.csv`).
3. `eda_seismic_noise.ipynb`: estacionariedad y elección de la banda del filtro.
4. `filters_design_freq.ipynb`: respuesta del filtro y comparación de bandas.
5. `deteccion_p.ipynb`: elección del largo de la LTA y del umbral del STA/LTA.
6. `prediccion_s.ipynb`: armado del dataset, regresión lineal y entrenamiento de la CNN (se guarda en `models/cnn.pt`).
7. `evaluacion_final.ipynb`: evaluación sobre test y figuras (se guardan en `docs/figures/`).

Para correrlos todos de una vez, con el entorno activado:

```
python main.py
```

Tarda alrededor de una hora en CPU, principalmente por el entrenamiento de la CNN. No se necesita GPU.

### 4. Reproducibilidad
La partición usa la semilla 0 y todas las muestras de trazas usan `random_state=0`, por lo que con los mismos chunks se obtienen exactamente las mismas trazas en cada conjunto. La CNN también se entrena con semilla fija; con otra versión de PyTorch u otro procesador los resultados pueden variar levemente en los últimos decimales.

## Autor
* Luca Emilio Petrarca - Proyecto Final para Procesamiento Avanzado de Señales - Facultad de Ciencias Exactas y Naturales UBA
