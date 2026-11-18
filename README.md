# Seismic Early Warning System using Advanced DSP and Deep Learning

## Descripción del Proyecto
Este proyecto desarrolla un pipeline de Procesamiento Digital de Señales (DSP) para la detección temprana de terremotos en entornos de alto ruido de fondo. Utilizando el Stanford Earthquake Dataset (STEAD), el sistema identifica ondas sísmicas primarias (Ondas P) ocultas en ruido microsísmico y antropogénico, para luego pronosticar la amplitud pico de la onda secundaria destructiva (Onda S) mediante modelos de Deep Learning.

## Objetivos Técnicos
* **Filtrado de Ruido:** Limpieza de la señal sísmica continua utilizando filtros pasabanda en el dominio de la frecuencia para mitigar el ruido oceánico y de tráfico.
* **Detección Automática:** Identificación del tiempo exacto de llegada de la Onda P mediante técnicas de correlación cruzada y *template matching*.
* **Forecasting Sísmico:** Comparación del rendimiento predictivo entre métodos de pronóstico tradicionales y redes neuronales (LSTM/CNN) evaluando una ventana de 3 segundos post-detección.

## Tecnologías Utilizadas
* **Lenguaje:** Python
* **Procesamiento de Señales:** SciPy, NumPy (Transformada de Fourier, Diseño de Filtros, Correlación)
* **Deep Learning / Forecasting:** PyTorch / TensorFlow Keras
* **Visualización:** Matplotlib, Seaborn (Espectrogramas, Análisis Tiempo-Frecuencia)

## Estructura del Repositorio
* `/src`: Scripts modulares con las funciones de filtrado, detección y predicción.
* `/notebooks`: Cuadernos de Jupyter con el análisis exploratorio y la justificación matemática.
* `/docs`: Documento final formato paper académico.
* `/data`: Carpeta ignorada en Git. Contiene los chunks del dataset `.hdf5`.
* `/metadata`: Metadata completa (CSV comprimido) de los chunks usados + `MANIFEST.md` con tamaños y conteos.
* `/scripts/download_stead.py`: Descarga y extrae los chunks a `data/raw/`.

## Datos
Fuente: **STEAD** (Mousavi et al., 2019, *IEEE Access*), https://github.com/smousavi05/STEAD, licencia CC BY 4.0.
Cada **traza** = registro de **una estación** durante **un evento** (sismo o ruido): 60 s a 100 Hz,
3 canales (E, N, Z) → array `(6000, 3)` `float32`. STEAD trae 6 chunks: 1 = ruido, 2–6 = sismos locales.

| chunk | contenido | trazas | eventos | redes / estaciones | M ≥ 4 | M ≥ 5 |
|---|---|---|---|---|---|---|
| 1 | ruido | 235 426 | – | 96 / 1150 | – | – |
| 2 | sismos locales | 200 000 | 101 395 | 43 / 202 (81 % red PB) | 2 932 | 243 |
| 6 | sismos locales | 230 231 | 137 645 | 100 / 881 | 7 668 | 637 |

**Qué se usa de cada chunk y por qué**
* **Detección de la onda P → chunks 1 + 2.** El 93 % de las trazas del chunk 2 son de estaciones que también
  tienen ruido en el chunk 1 (en el chunk 6, sólo el 34 %). Si sismo y ruido vienen de estaciones distintas, el
  detector puede aprender a reconocer la estación en vez del sismo. Del chunk 6 sólo se usan, para detección,
  las estaciones con ruido en el chunk 1.
* **Predicción de la amplitud de la S → chunks 2 + 6.** El chunk 2 sólo tiene 243 trazas de M ≥ 5 y está dominado
  por una red; sumando el 6 hay 10 600 trazas de M ≥ 4, 880 de M ≥ 5 y 109 redes.
* **Chunks 3, 4 y 5: no se usan.** Aportan menos diversidad y menos eventos grandes que el 6 (comparado con sus CSV).

**Reglas de partición**
* Train / validación / test se separan **por evento (`source_id`)**, nunca por traza ni por chunk:
  un mismo sismo aparece en varias trazas y en varios chunks (los chunks 2 y 6 comparten 19 873 eventos).

**Qué hay en el repo y qué no**
* **Sí:** la metadata **completa** (todas las filas y columnas) de los chunks 1, 2 y 6 en `metadata/chunkN.csv.gz`.
* **No:** las formas de onda (`chunkN.hdf5`, ~15 GB cada una). GitHub no admite archivos de más de 100 MB.
  Se bajan con `python scripts/download_stead.py`. No se recorta nada: el script baja los chunks originales.
* Los zip de STEAD tienen el índice ZIP64 roto (`unzip` y `zipfile` fallan); el script extrae leyendo los
  encabezados locales y verifica el CRC de cada archivo.

## Bitácora
* **15/11/2026** — Se descargaron los chunks 1 (ruido) y 6 (sismos) además del 2. Se eligió 1 + 2 + 6 por lo
  explicado en *Datos*. Se agregó la metadata al repo y el script de descarga.
* **17/11/2026** — Sanity check de los chunks 1, 2 y 6: filas iguales al manifest, P y S presentes y en orden.
  Split train/val/test (`src/data_loader.py`, scikit-learn): 70/10/20, semilla 0. Sismos separados por
  `source_id` con `GroupShuffleSplit` (219 167 eventos, ninguno en más de una parte); ruido con
  `train_test_split`, porque no tiene evento. val es para elegir umbrales y parámetros; test se usa una sola
  vez, al final. Se guarda en `data/processed/split.csv` (no versionado: se rearma igual con la misma semilla).

## Autor
* Luca Emilio Petrarca - Proyecto Final para Procesamiento Avanzado de Señales - Facultad de Ciencias Exactas y Naturales UBA
