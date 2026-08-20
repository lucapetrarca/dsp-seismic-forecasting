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

## Autor
* Luca Emilio Petrarca - Proyecto Final para Procesamiento Avanzado de Señales - Facultad de Ciencias Exactas y Naturales UBA
