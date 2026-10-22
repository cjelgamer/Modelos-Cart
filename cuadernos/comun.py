"""
Funciones y constantes compartidas por los cuadernos de la Tarea de ejecución 03.

Contiene:
  * Rutas del proyecto (relativas al archivo, nunca absolutas al equipo del autor).
  * Semilla global y helpers de reproducibilidad.
  * Carga y persistencia local de los conjuntos de datos.
  * Métricas de evaluación y utilidades de graficado con estilo uniforme.

Autor: Cristian Junior Cunurana Calderón
Curso: Aprendizaje de Máquina (UNAP)
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Rutas del proyecto
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "datos"
DIR_FIGURAS = RAIZ / "figuras"
DIR_RESULTADOS = RAIZ / "resultados"

for _d in (DIR_DATOS, DIR_FIGURAS, DIR_RESULTADOS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Reproducibilidad
# ---------------------------------------------------------------------------
SEMILLA = 42


def fijar_semilla(semilla: int = SEMILLA) -> None:
    """Fija todas las fuentes de aleatoriedad usadas en la práctica."""
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)


# ---------------------------------------------------------------------------
# 3. Paleta y estilo de graficado (coherente con la paleta del informe)
# ---------------------------------------------------------------------------
PALETA = {
    "azul": "#12355B",
    "azul_claro": "#2E6F95",
    "azul_profundo": "#0B1F33",
    "turquesa": "#1B998B",
    "verde": "#2A9D8F",
    "ambar": "#E9C46A",
    "naranja": "#F4A261",
    "rojo": "#C0392B",
    "gris": "#6C757D",
    "gris_claro": "#C9D1D9",
}


def aplicar_estilo() -> None:
    """Configura el estilo global de matplotlib para las figuras del informe."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": PALETA["azul_profundo"],
            "axes.labelcolor": PALETA["azul_profundo"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E3E7EA",
            "grid.linewidth": 0.7,
            "xtick.color": PALETA["azul_profundo"],
            "ytick.color": PALETA["azul_profundo"],
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    plt.rcParams["figure.autolayout"] = False


# ---------------------------------------------------------------------------
# 4. Datos
# ---------------------------------------------------------------------------
def cargar_breast_cancer():
    """Devuelve (X, y, nombres) del dataset breast_cancer incluido en sklearn.

    La clase positiva (1) es ``malignant`` (maligno); la negativa (0) es
    ``benign`` (benigno). La recodificación se hace aquí para que todos los
    cuadernos trabajen con la misma convención.
    """
    from sklearn.datasets import load_breast_cancer

    datos = load_breast_cancer(as_frame=False)
    X = datos.data
    y = datos.target  # 0 = benign, 1 = malignant
    nombres = list(datos.feature_names)
    assert set(np.unique(y)) == {0, 1}
    return X, y, nombres


URL_AUTO_MPG = "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip"


def _descargar_auto_mpg() -> Path:
    """Descarga el archivo original de la UCI y lo extrae en ``datos/``."""
    import io
    import urllib.request
    import zipfile

    destino = DIR_DATOS / "auto_mpg_uci.zip"
    print(f"Descargando dataset original desde UCI: {URL_AUTO_MPG}")
    with urllib.request.urlopen(URL_AUTO_MPG, timeout=120) as resp:
        contenido = resp.read()
    with zipfile.ZipFile(io.BytesIO(contenido)) as zf:
        crudo = zf.read("auto-mpg.data").decode("utf-8")
    destino.write_bytes(contenido)
    (DIR_DATOS / "auto_mpg_raw.txt").write_text(crudo, encoding="utf-8")
    return destino


def _parsear_auto_mpg() -> "object":
    """Transforma el archivo crudo de la UCI en un DataFrame con nombres claros."""
    import pandas as pd

    ruta_raw = DIR_DATOS / "auto_mpg_raw.txt"
    if not ruta_raw.exists():
        _descargar_auto_mpg()

    columnas = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]
    # El separador es espacio en blanco y el nombre del vehículo puede contener
    # espacios entre comillas; por eso se separa con un máximo de 8 cortes y se
    # reconstruye la última columna a partir del resto de la línea.
    filas = []
    patron = re.compile(r'^\s*([\d.?]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.?]+)\s+'
                         r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"?([^"]*)"?\s*$')
    for linea in ruta_raw.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        m = patron.match(linea)
        if m is None:
            raise ValueError(f"Línea no reconocida en auto-mpg.data: {linea!r}")
        valores = list(m.groups())
        filas.append(
            [
                float(valores[0]),
                int(float(valores[1])),
                float(valores[2]),
                np.nan if valores[3] == "?" else float(valores[3]),
                float(valores[4]),
                float(valores[5]),
                int(float(valores[6])),
                int(float(valores[7])),
                valores[8].strip(),
            ]
        )
    df = pd.DataFrame(filas, columns=columnas)
    # model_year viene como 70..82; se expresa como año calendario completo.
    df["model_year"] = df["model_year"] + 1900
    return df


def cargar_auto_mpg(descargar: bool = True):
    """Devuelve (df_completo, X, y) del dataset Auto MPG (UCI).

    ``descargar`` controla si se intenta recuperar el archivo de la UCI cuando
    la copia local ``datos/auto_mpg.csv`` no existe.
    """
    import pandas as pd

    ruta_csv = DIR_DATOS / "auto_mpg.csv"
    if ruta_csv.exists():
        df = pd.read_csv(ruta_csv)
    elif descargar:
        df = _parsear_auto_mpg()
        df.to_csv(ruta_csv, index=False)
    else:
        raise FileNotFoundError(
            f"No se encontró {ruta_csv}. Ejecute el cuaderno 01 con descarga habilitada."
        )

    X = df.drop(columns=["mpg", "car_name"])
    y = df["mpg"]
    return df, X, y


# ---------------------------------------------------------------------------
# 5. Persistencia de resultados
# ---------------------------------------------------------------------------
def guardar_json(datos: dict, nombre: str) -> Path:
    """Guarda un diccionario de resultados como JSON legible en ``resultados/``."""
    ruta = DIR_RESULTADOS / nombre
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(datos, fh, indent=2, ensure_ascii=False, default=str)
    return ruta


def guardar_csv(df, nombre: str) -> Path:
    """Guarda un DataFrame como CSV en ``resultados/``."""
    ruta = DIR_RESULTADOS / nombre
    df.to_csv(ruta, index=False, encoding="utf-8")
    return ruta
