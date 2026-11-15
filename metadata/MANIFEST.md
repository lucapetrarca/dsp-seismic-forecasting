# Manifest de datos (STEAD)

Archivos en `data/raw/` (no versionados). Verificados el 15/11/2026: CRC del zip ok, misma cantidad de trazas
en el `.hdf5` (grupo `data/`) que filas en el `.csv`, cada traza `(6000, 3)` `float32`.

| chunk | contenido | trazas | chunkN.csv (bytes) | sha256 del csv | chunkN.hdf5 (bytes) |
|---|---|---|---|---|---|
| 1 | ruido | 235 426 | 26 769 158 | `d631967e4ed717df…` | 17 880 084 720 |
| 2 | sismos locales | 200 000 | 67 191 714 | `ee6a01f738ef0f91…` | 15 159 483 640 |
| 6 | sismos locales | 230 231 | 76 261 049 | `f3943dba582baae1…` | 17 448 081 440 |

`metadata/chunkN.csv.gz` = el `chunkN.csv` original completo, comprimido con gzip (sin cambios).
Leer con `pd.read_csv("metadata/chunk2.csv.gz", low_memory=False)`.
