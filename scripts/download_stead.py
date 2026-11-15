"""Descarga y extrae los chunks de STEAD que usa el proyecto (1 = ruido, 2 y 6 = sismos locales).

Uso:
    python scripts/download_stead.py            # chunks 1, 2 y 6 -> data/raw/
    python scripts/download_stead.py 6          # sólo el chunk 6

Notas:
- Los links son los oficiales del repo de STEAD (https://github.com/smousavi05/STEAD), que redirigen a Google Drive.
- Los .zip de STEAD superan 4 GB y tienen el índice central ZIP64 mal escrito: `unzip` y `zipfile`
  fallan ("Bad magic number"). Por eso se extrae leyendo los encabezados locales en orden y se
  verifica el CRC32 de cada archivo.
- La descarga se puede retomar: si se corta, volver a correr el script.
"""
import os
import struct
import sys
import urllib.request
import zlib

DRIVE_IDS = {
    1: "1sgLd3eVIrTMcuWtOQRN1fGmiNQ7vc227",  # ruido, ~14.6 GB
    2: "1ONNNtiOwFodAAeV-ZWp_-gtWhRcYgU_E",  # sismos locales, ~13.7 GB
    3: "15bt3WS-mQJmkVY4VOb5LYk45xQBYtkoM",
    4: "12Rw-WI9T21g0hrqtF1KiETYzV4rz177k",
    5: "1rR6bbqu1yE8N_EVIb3kH3wzPPVLFerFX",
    6: "1QfONZf8TBuQlkvKjwJgVMgZub4RDUkH9",  # sismos locales, ~15.7 GB
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CHUNK = 16 << 20


def url(fid):
    return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"


def download(fid, dst):
    """Descarga con reanudación (HTTP Range)."""
    done = os.path.getsize(dst) if os.path.exists(dst) else 0
    req = urllib.request.Request(url(fid), headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={done}-"})
    with urllib.request.urlopen(req) as r, open(dst, "ab") as f:
        total = done + int(r.headers.get("Content-Length", 0))
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            f.write(b)
            done += len(b)
            print(f"\r  {done / 2**30:6.2f} / {total / 2**30:.2f} GB", end="", flush=True)
    print()


def extract(zip_path, out_dir):
    """Extrae todas las entradas deflate leyendo encabezados locales en secuencia."""
    f = open(zip_path, "rb")
    off = 0
    while True:
        f.seek(off)
        h = f.read(30)
        if h[:4] != b"PK\x03\x04":
            break
        flag, meth = struct.unpack("<HH", h[6:10])
        nl, el = struct.unpack("<HH", h[26:30])
        name = f.read(nl).decode()
        f.read(el)
        assert meth == 8, f"{name}: método de compresión {meth} no soportado"
        d, crc, size = zlib.decompressobj(-15), 0, 0
        part = os.path.join(out_dir, name + ".part")
        with open(part, "wb") as o:
            while not d.eof:
                c = f.read(CHUNK)
                if not c:
                    raise SystemExit(f"{name}: zip truncado, volver a descargar")
                b = d.decompress(c)
                o.write(b)
                crc = zlib.crc32(b, crc)
                size += len(b)
        # posición absoluta donde terminó el bloque comprimido
        end = f.tell() - len(d.unused_data)
        f.seek(end)
        tail = f.read(64)
        if flag & 8:  # data descriptor: [PK\x07\x08] crc32 ...
            i = 4 if tail[:4] == b"PK\x07\x08" else 0
            expected = struct.unpack("<I", tail[i:i + 4])[0]
        else:
            expected = struct.unpack("<I", h[14:18])[0]
        if crc != expected:
            raise SystemExit(f"{name}: CRC no coincide ({crc:08x} vs {expected:08x})")
        os.replace(part, os.path.join(out_dir, name))
        print(f"  {name}: {size / 2**30:.2f} GB, CRC ok")
        nxt = tail.find(b"PK\x03\x04")
        if nxt < 0:
            break
        off = end + nxt


def main(chunks):
    os.makedirs(RAW, exist_ok=True)
    for n in chunks:
        if os.path.exists(os.path.join(RAW, f"chunk{n}.hdf5")):
            print(f"chunk{n}: ya está en data/raw, salteo")
            continue
        z = os.path.join(RAW, f"chunk{n}.zip")
        print(f"chunk{n}: descargando")
        download(DRIVE_IDS[n], z)
        print(f"chunk{n}: extrayendo")
        extract(z, RAW)
        os.remove(z)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 6])
