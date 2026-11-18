# Optimización de hiperparámetros · experimento 2

Mismo método, mismos archivos y mismas gráficas que el experimento 1. Cómo se organiza cada archivo y cómo leer cada gráfica está en el [README del experimento 1](../04-hyperparameter-optimization-experiment-1/README.md).

## Qué cambia

El experimento 2 usa otras variables, así que la búsqueda se repite: los mejores ajustes dependen de las variables de entrada. Los cambios y su motivo están explicados en los notebooks de modelado: [SVM](../04-modeling-experiment-2/04.2-svm.ipynb) y [KNN](../04-modeling-experiment-2/04.2-knn.ipynb).

| | Experimento 1 | Experimento 2 |
|---|---|---|
| Variables en SVM y KNN | 19 | 22: se agregan raza, color y ciudad |
| Edad en regresión | `age_days` | `age_log` (logaritmo de la edad) |
| Rango de `epsilon` (regresión, solo SVM) | 0,01 a 0,5 | 0,05 a 1 |

## Modelos

SVM y KNN, con las mismas variables para poder compararlos. Los archivos se llaman igual que en el experimento 1 (`svm_clf`, `svm_reg`, `knn_clf`, `knn_reg`), pero cada experimento guarda los suyos en su propia carpeta, así que no se pisan.

Si otro modelo se repite en el experimento 2, debería usar las mismas variables para poder compararlo.
