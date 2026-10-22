# Modelos CART y ensamblados — Tarea de ejecución 03

**Aprendizaje de Máquina — Universidad Nacional del Altiplano (UNAP)**
Estudiante: Cristian Junior Cunurana Calderon · Escuela Profesional de Ingeniería de Sistemas · Ciclo IX · Grupo B

Modelos de árboles de decisión (CART), bagging, RandomForest y XGBoost aplicados a los conjuntos
`breast_cancer` (clasificación) y `auto_mpg` (regresión).

- **Informe final compilado:** [`Informe_Tarea_03.pdf`](Informe_Tarea_03.pdf) (30 páginas, 25 figuras, 15 tablas)
- **Código fuente del informe:** [`main.tex`](main.tex) (LaTeX, `latexmk -pdf main.tex`)
- **Cuadernos ejecutados (sin errores):** `cuadernos/01_exploracion_datos.ipynb`, `cuadernos/02_arboles_y_modelos_lineales.ipynb`, `cuadernos/03_ensamblados.ipynb`

---

## 1. Metodología común

Todos los cuadernos comparten el módulo `cuadernos/comun.py`, de modo que el protocolo es
idéntico en los cinco ejercicios:

| Aspecto | Decisión adoptada |
|---|---|
| Partición | 80 % entrenamiento / 20 % prueba, estratificada por clase en clasificación, semilla 42 |
| Validación | `StratifiedKFold` / `KFold` con 5 particiones, `shuffle=True`, semilla 42 |
| Selección | `GridSearchCV` ejecutado **solo** sobre el conjunto de entrenamiento |
| Preprocesamiento | Dentro de `Pipeline`; ajustado únicamente con datos de entrenamiento |
| Métricas (clasificación) | Exactitud, precisión, exhaustividad, F1 y AUC-ROC (positiva = `malignant`) |
| Métricas (regresión) | RMSE, MAE y $R^2$ |
| Estabilidad | 7 particiones 80/20 con semillas 42, 7, 13, 21, 99, 123 y 2026 |
| Referencia base | Modelo `dummy` (clase mayoritaria en clasificación, media en regresión) |

Ningún resultado proviene de una ejecución con errores: los cuadernos se ejecutan con
`nbconvert --execute` **sin** `--allow-errors`.

---

## 2. Estructura del informe

Portada · Índice general · **Resumen** · **1. Link del repositorio** (enlace al código, estructura
del repositorio, flujo de trabajo y comandos de reproducción) · **2. Datos y preparación
experimental** · **3.–7. Los cinco ejercicios** (uno por capítulo, con configuración, resultados y
respuesta razonada) · **8. Conclusiones** (respuestas, conclusiones finales y límites de validez) ·
**Anexo A. Tablas completas de validación cruzada** (con la tabla de modelos comparados).

## 3. Respuestas a las preguntas del enunciado

### Pregunta 1 — ¿Qué modelo es más eficiente en *breast_cancer*: Gini, entropía o regresión logística?

**La regresión logística es la más eficiente.** F1 de prueba 0,9793 frente a 0,9517 (Gini) y 0,8971
(entropía); F1 de validación cruzada 0,9861 ± 0,0042 frente a 0,9460 ± 0,0124 (Gini) y
0,9509 ± 0,0154 (entropía); AUC-ROC de prueba 0,9957 frente a 0,9439 y 0,9593.

La entropía **sí** ordena mejor las probabilidades que el Gini (AUC 0,9593 frente a 0,9439), pero su
punto de operación sobre el umbral 0,5 es peor (exactitud 0,8772 frente a 0,9386). En los dos
árboles, además, aumentar la profundidad no mejora el F1 de validación cruzada (0,9460 con
profundidad 3 frente a 0,9425 sin límite de profundidad), señal de sobreajuste.

### Pregunta 2 — ¿Qué árbol de regresión supera al regresor lineal en *auto_mpg*?

**El árbol de regresión con criterio `absolute_error`, profundidad 6 y 10 observaciones por hoja.**
RMSE de prueba 2,5700 frente a 2,8877 mpg (11,00 % menos), MAE 1,8850 frente a 2,2882 mpg y
$R^2$ 0,8772 frente a 0,8449; 112 configuraciones evaluadas. La regresión lineal solo es más
estable entre particiones (RMSE 3,3506 ± 0,3734 frente a 3,0976 ± 0,6492 del árbol).

### Pregunta 3 — ¿Los ensembles superan al mejor modelo individual?

**Sí, pero no al mejor modelo individual en *breast_cancer*.** El mejor ensemble es el
RandomForest (F1 de prueba 0,9655, AUC 0,9931), seguido de GradientBoosting (0,9589) y la
votación blanda (0,9583); el mejor modelo individual es la regresión logística (F1 0,9793,
AUC 0,9957). Los ensembles reducen el sobreajuste de los árboles (el árbol Gini sube de 0,9460 a
0,9700 en validación cruzada), pero no alcanzan a un modelo de pesos continuos bien
regularizado.

### Pregunta 4 — ¿Qué aporta el bagging?

**Reduce la varianza del árbol individual sin perder interpretabilidad.**
En regresión, el bagging (200 árboles, `max_samples=1,0`) alcanza RMSE 2,4901 y $R^2$ 0,8847
frente a 2,5700 y 0,8772 del árbol individual, pero el RandomForest lo mejora hasta 2,1580 y
$R^2$ 0,9134. En clasificación, el bagging sube el AUC-ROC de 0,9439 a 0,9917 y el F1 de 0,9517
a 0,9583, sin alcanzar el 0,9793 de la regresión logística.

### Pregunta 5 — ¿RandomForest o XGBoost?

**Empate técnico; XGBoost es más preciso en validación cruzada y algo más rápido.**
F1 de validación cruzada 0,9807 ± 0,0129 (XGBoost) frente a 0,9719 ± 0,0116 (RandomForest), con
AUC 0,9939 frente a 0,9892. En el conjunto de prueba la diferencia es de 0,0005 en F1
(0,9660 frente a 0,9655), inferior a la desviación estándar de la validación cruzada, por lo que
el punto de prueba no permite separar los modelos. XGBoost tarda 12,18 s frente a 15,05 s de
entrenamiento. Las dos familias coinciden en las variables más informativas: radio máximo,
perímetro y puntos de concavidad.

---

## 4. Estructura del repositorio

```
Modelos-Cart/
├── Informe_Tarea_03.pdf      Informe final compilado (30 páginas)
├── main.tex                  Código LaTeX del informe
├── README.md                 Este documento
├── requirements.txt          Versiones exactas del entorno
├── cuadernos/
│   ├── comun.py              Módulo compartido (rutas, semilla, estilo, carga y guardado)
│   ├── 01_exploracion_datos.ipynb    Ejercicio 0: exploración y diagnóstico (figuras 2.1–2.7)
│   ├── 02_arboles_y_modelos_lineales.ipynb  Ejercicios 1 y 2 (figuras 3.1–4.4)
│   └── 03_ensamblados.ipynb  Ejercicios 3, 4 y 5 (figuras 5.1–7.3)
├── datos/                    Copia local de auto_mpg (CSV, texto original y ZIP de la UCI)
├── figuras/                  Las 24 figuras generadas por los cuadernos (PNG)
└── resultados/               Métricas de todos los experimentos (CSV) y resúmenes (JSON)
```

---

## 5. Reproducción del trabajo

```bash
# 1. Entorno
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Kernel con el nombre usado en el informe
python -m ipykernel install --user --name aprendizaje-maquina-t3 \
    --display-name "Python 3.14 (Aprendizaje Maquina T3)"

# 3. Ejecución de los tres cuadernos (debe terminar sin errores)
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=aprendizaje-maquina-t3 \
    cuadernos/01_exploracion_datos.ipynb
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=aprendizaje-maquina-t3 \
    cuadernos/02_arboles_y_modelos_lineales.ipynb
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=aprendizaje-maquina-t3 \
    cuadernos/03_ensamblados.ipynb

# 4. Compilación del informe
latexmk -pdf -interaction=nonstopmode main.tex
```

Los cuadernos escriben las figuras en `figuras/` y las métricas en `resultados/`, por lo que el
informe puede recompilarse sin volver a ejecutar los experimentos.

---

## 6. Entorno computacional

| Componente | Versión |
|---|---|
| Python | 3.14.8 |
| NumPy | 2.5.4 |
| Pandas | 3.0.6 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| Matplotlib | 3.11.2 |
| Seaborn | 0.13.2 |
| JupyterLab | 4.6.4 |
| ipykernel | 7.4.0 |

Semilla global 42 aplicada a `random`, NumPy, scikit-learn y XGBoost.
