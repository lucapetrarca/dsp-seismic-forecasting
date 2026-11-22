#Predicción de la amplitud de la onda S a partir de los 3 s que siguen a la P. Se comparan una regresión lineal y una CNN.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal
from src.data_loader import RAIZ, PROCESSED, cargar
from src.dsp_filters import FS, filtrar

VENTANA = 3 * FS  #3 s desde la P
VENTANA_S = 5 * FS  #5 s desde la S
RUTA_CNN = RAIZ / 'models' / 'cnn.pt'

def ventana(trf, p):
    #Se recortan los 3 s desde la P de la traza filtrada, queda (3, 300) con los 3 canales.
    return trf[p:p + VENTANA].T

def armar_dataset(tabla, nombre):
    #Arma X, y y los datos de cada sismo, y lo guarda en data/processed/<nombre>.npz y <nombre>_meta.csv.
    ruta, ruta_meta = PROCESSED / f'{nombre}.npz', PROCESSED / f'{nombre}_meta.csv'
    if ruta.exists() and ruta_meta.exists():
        d = np.load(ruta)
        return d['X'], d['y'], pd.read_csv(ruta_meta, dtype={'evento': str})
    #Se descartan los sismos con la S dentro de la ventana, dado que si no el modelo vería la respuesta.
    p = tabla['p_arrival_sample'].astype(int)
    s = tabla['s_arrival_sample'].astype(int)
    tabla = tabla[(s - p > VENTANA) & (s + VENTANA_S <= 6000)]
    X, y, quedan = [], [], []
    for i, fila in enumerate(tabla.itertuples()):
        tr = cargar(fila.trace_name, fila.chunk).astype(float)
        #Se descartan las trazas con algún canal con desvío menor a 0.1.
        if (tr.std(axis=0) < 0.1).any():
            continue
        p, s = int(fila.p_arrival_sample), int(fila.s_arrival_sample)
        X.append(ventana(filtrar(tr), p))
        #El pico de la S se mide en las horizontales (E y N) sin el filtro de 3-20 Hz, para no recortar la amplitud real.
        y.append(np.log10(np.abs(tr[s:s + VENTANA_S, :2]).max() + 1e-9))
        quedan.append(i)
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    tabla = tabla.iloc[quedan]
    #Datos de cada sismo que no entran al modelo, se usan para evaluar por grupos.
    meta = pd.DataFrame({
        'trace_name': tabla['trace_name'].to_numpy(),
        'chunk': tabla['chunk'].to_numpy(),
        'p': tabla['p_arrival_sample'].astype(int).to_numpy(),
        'evento': tabla['source_id'].to_numpy(),
        'mag': tabla['source_magnitude'].to_numpy(),
        'dist_km': tabla['source_distance_km'].to_numpy(),
        #snr_db trae la SNR de los tres canales como texto, se toma la del canal Z.
        'snr': tabla['snr_db'].str.strip('[]').str.split().str[2].astype(float).to_numpy(),
    })
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez(ruta, X=X, y=y)
    meta.to_csv(ruta_meta, index=False)
    return X, y, meta

def features(w):
    #Features de alerta temprana sobre la ventana de la P (3, 300).
    z = w[2]
    #Se integra la velocidad para tener el desplazamiento
    d = np.cumsum(z) / FS
    v2, d2 = np.sum(np.diff(z) ** 2), np.sum(np.diff(d) ** 2)
    #tau_c: período característico, los sismos grandes tienen más energía en bajas frecuencias.
    tau_c = 2 * np.pi * np.sqrt(d2 / (v2 + 1e-20))
    f, P = signal.welch(z, fs=FS, nperseg=128)
    return [
        np.log10(np.abs(w).max() + 1e-9),  #Pico
        np.log10(np.sqrt(np.mean(w ** 2)) + 1e-9),  #RMS
        np.log10(np.abs(d).max() + 1e-9),  #Pd, pico de desplazamiento
        tau_c,
        f[np.argmax(P)],  #Frecuencia dominante
    ]

def separar_escala(X):
    #Se separa la forma (ventana dividida por su máximo) de la escala (log10 del máximo).
    #La amplitud es lo que más informa sobre la S.
    escala = np.log10(np.abs(X).max(axis=(1, 2)) + 1e-9)
    forma = X / (10 ** escala)[:, None, None]
    return forma.astype(np.float32), escala.astype(np.float32)

class CNN(nn.Module):
    #Tres capas convolucionales miran la forma de la P y la escala se suma antes de la salida.
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.salida = nn.Sequential(nn.Linear(64 + 1, 32), nn.ReLU(), nn.Linear(32, 1))
        #Media y desvío de la escala en train, se guardan con el modelo.
        self.register_buffer('esc_media', torch.tensor(0.0))
        self.register_buffer('esc_desvio', torch.tensor(1.0))

    def forward(self, forma, escala):
        h = self.conv(forma).squeeze(-1)
        e = (escala - self.esc_media) / self.esc_desvio
        return self.salida(torch.cat([h, e[:, None]], dim=1)).squeeze(-1)

def predecir_cnn(modelo, X):
    forma, escala = separar_escala(X)
    modelo.eval()
    with torch.no_grad():
        return modelo(torch.from_numpy(forma), torch.from_numpy(escala)).numpy()

def entrenar_cnn(X_tr, y_tr, X_va, y_va, epocas=20, semilla=0):
    #Se entrena con MAE como pérdida y se guarda la época con menor error en validación.
    torch.manual_seed(semilla)
    forma, escala = separar_escala(X_tr)
    modelo = CNN()
    modelo.esc_media.fill_(float(escala.mean()))
    modelo.esc_desvio.fill_(float(escala.std()))
    #La salida arranca en el promedio de "y" para no perder las primeras épocas.
    modelo.salida[-1].bias.data.fill_(float(y_tr.mean()))
    datos = torch.utils.data.TensorDataset(torch.from_numpy(forma), torch.from_numpy(escala), torch.from_numpy(y_tr))
    lotes = torch.utils.data.DataLoader(datos, batch_size=256, shuffle=True)
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    perdida = nn.L1Loss()
    historia, mejor = [], np.inf
    RUTA_CNN.parent.mkdir(parents=True, exist_ok=True)
    for epoca in range(epocas):
        modelo.train()
        suma = 0.0
        for fb, eb, yb in lotes:
            opt.zero_grad()
            loss = perdida(modelo(fb, eb), yb)
            loss.backward()
            opt.step()
            suma += loss.item() * len(yb)
        mae_va = np.mean(np.abs(predecir_cnn(modelo, X_va) - y_va))
        historia.append({'epoca': epoca, 'mae_train': suma / len(y_tr), 'mae_val': mae_va})
        if mae_va < mejor:
            mejor = mae_va
            torch.save(modelo.state_dict(), RUTA_CNN)
    modelo.load_state_dict(torch.load(RUTA_CNN))
    return modelo, pd.DataFrame(historia)

def cargar_cnn():
    modelo = CNN()
    modelo.load_state_dict(torch.load(RUTA_CNN))
    return modelo