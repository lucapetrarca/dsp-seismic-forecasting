# Seismic Early Warning System using Advanced DSP and Deep Learning

## Descripción del Proyecto
Este proyecto desarrolla un pipeline de Procesamiento Digital de Señales (DSP) para la detección temprana de terremotos en entornos de alto ruido de fondo. Utilizando el Stanford Earthquake Dataset (STEAD), el sistema identifica ondas sísmicas primarias (Ondas P) ocultas en ruido microsísmico y antropogénico, para luego pronosticar la amplitud pico de la onda secundaria destructiva (Onda S) mediante modelos de Deep Learning.

## Objetivos Técnicos
* **Filtrado de Ruido:** Limpieza de la señal sísmica continua utilizando filtros pasabanda en el dominio de la frecuencia para mitigar el ruido oceánico y de tráfico.
* **Detección Automática:** Identificación del tiempo exacto de llegada de la Onda P mediante el detector STA/LTA.
* **Forecasting Sísmico:** Comparación del rendimiento predictivo entre una regresión lineal con features clásicas y una red neuronal convolucional (CNN 1D) evaluando una ventana de 3 segundos post-detección.

## Tecnologías Utilizadas
* **Lenguaje:** Python
* **Procesamiento de Señales:** SciPy, NumPy (Transformada de Fourier, Diseño de Filtros)
* **Deep Learning / Forecasting:** PyTorch, scikit-learn
* **Visualización:** Matplotlib

## Estructura del Repositorio
* `/src`: Scripts modulares con las funciones de filtrado, detección y predicción.
* `/notebooks`: Cuadernos de Jupyter con el análisis exploratorio y la justificación matemática.
* `/docs/figures`: Figuras de cada etapa y de la evaluación final.
* `/models/cnn.pt`: Pesos de la CNN entrenada.
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
  las estaciones con ruido en el chunk 1. Como finalmente el detector es un STA/LTA, que no se entrena (sólo se
  elige el umbral), se evaluó con sismos de los chunks 2 y 6.
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

## Orden de los notebooks
1. `sanity_check.ipynb` y `carga_y_split.ipynb`: revisión de los datos y split train/val/test.
2. `eda_seismic_noise.ipynb`: estacionariedad y elección de la banda del filtro.
3. `filters_design_freq.ipynb`: respuesta del filtro y SNR con distintas bandas.
4. `deteccion_p.ipynb`: STA/LTA y elección del umbral en validación.
5. `prediccion_s.ipynb`: dataset, regresión lineal y CNN en validación.
6. `evaluacion_final.ipynb`: todo en test, pipeline completo y espectrograma.

Para correr todos en orden: `python main.py` (con el entorno virtual activado y los chunks en `data/raw/`).

## Resultados (test)
* **Detección de la P (STA/LTA, LTA 5 s, umbral 5.5):** 82.6 % de los sismos detectados a menos de 0.5 s de la P,
  con 5.2 % de falsas alarmas en ruido. Por SNR: 57 % con menos de 10 dB y 98 % con más de 30 dB.
* **Amplitud de la S (MAE en log10):** promedio 0.747, regresión lineal 0.227 (factor 1.69), CNN 0.206 (factor 1.61).
  Con magnitud mayor a 4: lineal 0.341, CNN 0.295.
* **Pipeline completo:** con la ventana desde el pick del STA/LTA la CNN da 0.215 contra 0.209 con la P de STEAD;
  el 8.4 % de los sismos queda sin pick.

## Bitácora
* **15/11/2026** — Se descargaron los chunks 1 (ruido) y 6 (sismos) además del 2. Se eligió 1 + 2 + 6 por lo
  explicado en *Datos*. Se agregó la metadata al repo y el script de descarga.
* **17/11/2026** — Sanity check de los chunks 1, 2 y 6: filas iguales al manifest, P y S presentes y en orden.
  Split train/val/test (`src/data_loader.py`, scikit-learn): 70/10/20, semilla 0. Sismos separados por
  `source_id` con `GroupShuffleSplit` (219 167 eventos, ninguno en más de una parte); ruido con
  `train_test_split`, porque no tiene evento. val es para elegir umbrales y parámetros; test se usa una sola
  vez, al final. Se guarda en `data/processed/split.csv` (no versionado: se rearma igual con la misma semilla).
* **19/11/2026** — Análisis exploratorio en `notebooks/eda_seismic_noise.ipynb`, sólo con trazas de train y el
  canal Z. Sobre 300 trazas de cada tipo, ADF y KPSS indicaron que tanto el ruido como los sismos son
  estacionarios, ya que esos tests evalúan la media y en STEAD la media es siempre cero. Lo que cambia en un sismo
  es la varianza, por lo que se agregó el cociente entre el mayor y el menor desvío en ventanas de 5 s: supera 3 en
  el 9 % del ruido y en el 94 % de los sismos. Se encontró que una parte chica del ruido tiene el canal Z muerto
  (desvío casi nulo, cerca del 0.3 %), y esas trazas se descartan. La PSD del ruido muestra que STEAD ya viene
  filtrado entre 1 y 45 Hz. Comparando la PSD mediana de 491 sismos antes y después de la P, el sismo supera al
  ruido por 10 dB entre 1 y 45 Hz (la banda que ya trae STEAD) y por 20 dB entre 3 y 22 Hz. Se eligió un
  Butterworth pasabanda de 3 a 20 Hz, orden 4 y causal (`src/dsp_filters.py`), que lleva la SNR mediana de 19 a
  24 dB. Una banda de 5 a 15 Hz da algo más (25.6 dB), pero recorta más la señal de los sismos grandes.
  La banda se verificó también en los canales E y N (2.7 a 22.7 Hz con 20 dB), por lo que sirve para los tres.
  El diseño y la prueba del filtro se pasaron a `notebooks/filters_design_freq.ipynb`.
* **20/11/2026** — Detección de la onda P (`src/detection.py`, `notebooks/deteccion_p.ipynb`), con el
  umbral elegido sobre validación y sin usar test. Se implementó un STA/LTA causal sobre el canal Z filtrado,
  con STA de 0.5 s. Se probaron LTA de 3, 5 y 10 s; con 10 s el resultado empeora porque la P llega en promedio a
  los 7 s de la traza y la LTA no alcanza a llenarse de ruido, por lo que se eligió 5 s. Un pick se consideró
  correcto si cae a menos de 0.5 s de la P marcada, y una falsa alarma es cualquier disparo en una traza de ruido.
  Con el umbral 5.5, el más bajo que da a lo sumo 5 % de falsas alarmas, se detectó el 83 % de 2000 sismos,
  bajando al 54 % con SNR menor a 10 dB. El pick llega en general entre 0 y 0.2 s después de la P. También se
  probó template matching, con las 5 P de mejor SNR de train de cada estación como plantillas, pero detectó sólo
  el 1.5 % con el mismo criterio de falsas alarmas, ya que las P de distintos sismos tienen formas distintas. Por
  eso se descartó y el detector del proyecto es el STA/LTA.
* **21/11/2026** — Predicción de la amplitud de la S (`src/forecasting.py`, `notebooks/prediccion_s.ipynb`). X son
  los 3 s desde la P de STEAD en la traza filtrada (3 canales) e y es el log10 del pico de E y N en los 5 s desde la
  S, sin filtrar para no recortar la amplitud real. Se descartaron los sismos con la S a menos de 3 s de la P
  (quedaría dentro de la ventana) y los que tienen algún canal muerto, y se usó una muestra de 60 000 sismos de
  train y 10 000 de validación para entrenar en CPU. El baseline es una regresión lineal con pico, RMS, Pd, τc y
  frecuencia dominante; la CNN tiene tres capas convolucionales sobre la ventana normalizada y recibe aparte el
  log10 del máximo, para no perder la amplitud. Se entrenó 20 épocas con MAE como pérdida, guardando la de menor
  error en validación. En validación: lineal 0.229 y CNN 0.207.
  Evaluación final (`notebooks/evaluacion_final.ipynb`), única vez que se usó test: los resultados están en
  *Resultados*. Las figuras quedaron en `docs/figures/`.

## Autor
* Luca Emilio Petrarca - Proyecto Final para Procesamiento Avanzado de Señales - Facultad de Ciencias Exactas y Naturales UBA
