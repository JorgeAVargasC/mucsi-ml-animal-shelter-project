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

### Árbol de decisión y random forest

También se repiten en el experimento 2 (`tree_clf`, `tree_reg`, `rf_clf`, `rf_reg`), con cambios propios. Los motivos están en los notebooks de modelado: [árbol de decisión](../04-modeling-experiment-2/04.2-decision-tree.ipynb) y [random forest](../04-modeling-experiment-2/04.2-random-forest.ipynb).

| | Experimento 1 | Experimento 2 |
|---|---|---|
| Variables | las 22 | las mismas 22 |
| Raza, color y ciudad | one-hot; las categorías con menos del 1 % se juntan | *target encoding*: cada categoría se cambia por el promedio del objetivo y no se pierde ninguna |
| `max_depth` | árbol: 2 a 30; bosque: 5 a 40 | árbol: 2 a 60; bosque: 5 a 60 |
| `max_features` (solo bosque) | 0,05 a 0,6 | 0,05 a 0,8 |
| `criterion` (solo clasificación) | `gini`, sin buscar | `gini` o `entropy` |
| `class_weight` (solo bosque) | `None` o `balanced` | se agrega `balanced_subsample` |

La edad sigue en `age_days`: un árbol corta igual por la edad que por su logaritmo, así que `age_log` no cambia nada.
