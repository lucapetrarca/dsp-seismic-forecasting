#Detección de la onda P con STA/LTA. El umbral fue elegido con validación en notebooks/deteccion_p.ipynb.
import numpy as np
from scipy import signal
from src.dsp_filters import FS

#Ventanas del STA/LTA en muestras, 0.5 s y 5 s.
STA = int(0.5 * FS)
LTA = int(5 * FS)
UMBRAL_STA_LTA = 5.5

def media_exp(e, n):
    #Promedio causal de los últimos n valores con peso exponencial.
    #Cuando todavía no hay n muestras se usa el promedio de las anteriores.
    e = np.asarray(e, dtype=float)
    m = np.empty_like(e)
    k = min(n, len(e))
    m[:k] = np.cumsum(e[:k]) / np.arange(1, k + 1)
    if len(e) > n:
        c = 1 / n
        m[n:], _ = signal.lfilter([c], [1, c - 1], e[n:], zi=[(1 - c) * m[n - 1]])
    return m

def sta_lta(zf, sta=STA, lta=LTA):
    #Cociente entre la energía reciente y la energía de fondo. La señal tiene que estar filtrada.
    e = np.asarray(zf, dtype=float) ** 2
    return media_exp(e, sta) / (media_exp(e, lta) + 1e-12)

def primer_cruce(c, umbral, desde=FS):
    #Primera muestra en la que c supera el umbral, si nunca lo supera devuelve None.
    #Se saltea el primer segundo, donde el filtro y los promedios todavía no se estabilizaron.
    idx = np.flatnonzero(c[desde:] > umbral)
    return int(idx[0]) + desde if len(idx) else None