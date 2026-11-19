#Filtro pasabanda. La banda fue elegida en notebooks/eda_seismic_noise.ipynb.
import numpy as np
from scipy import signal

FS = 100  #Muestras por segundo en el dataset.

#Banda de 3-20Hz, orden 4.
BANDA = (3, 20)
ORDEN = 4

#Se arma el Butterworth en formato sos, dado que es más estable que usar b y a
SOS = signal.butter(ORDEN, BANDA, btype='bandpass', fs=FS, output='sos')

def filtrar(x):
    #Se filtra a lo largo del tiempo. Sosfilt es causal, es decir usa muestras pasadas únicamente.
    return signal.sosfilt(SOS, np.asarray(x, dtype=float), axis=0)