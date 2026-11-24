# Optimización de hiperparámetros · experimento 1

Cada modelo tiene "ajustes" que no se aprenden de los datos y hay que elegir antes de entrenar (por ejemplo, qué tan estricto es el modelo con los errores). Aquí se buscan con **Optuna**: prueba muchas combinaciones y se queda con la mejor. El notebook de modelado del mismo experimento usa esa combinación.

## Qué se guarda

Cada búsqueda deja tres tipos de archivos. Sus nombres siguen el patrón `<modelo>_<tarea>`, donde `clf` es clasificación y `reg` es regresión. El nombre lo define cada notebook en el primer argumento de `tune(...)`, y el de modelado lo usa en `load_best(...)`. Estos son los nombres acordados:

| Modelo | Nombres de los archivos |
|---|---|
| Regresión lineal y logística | `linear_clf`, `linear_reg` |
| KNN | `knn_clf`, `knn_reg` |
| Naive Bayes | `nb_clf` (solo clasifica) |
| SVM | `svm_clf`, `svm_reg` |
| Árbol de decisión | `tree_clf`, `tree_reg` |
| Random forest | `rf_clf`, `rf_reg` |

SVM es solo un ejemplo de algoritmo: cada modelo guarda sus propios archivos, con su prefijo. Por ejemplo, la búsqueda de SVM para clasificación genera:

```
best_params/svm_clf.json      la mejor combinación y cómo se obtuvo
trials/svm_clf.csv            todas las pruebas que se hicieron
figures/svm_clf_historial.png
figures/svm_clf_cortes.png
```

En el resto de este documento, `<modelo>_<tarea>` es cualquiera de esos nombres.

### `best_params/<modelo>_<tarea>.json`: la mejor combinación

| Campo | Qué significa |
|---|---|
| `mejores_parametros` | Los ajustes ganadores. Son los que usa el modelado. |
| `significado_de_cada_ajuste` | Qué hace cada ajuste, en una frase. |
| `resultado_de_la_busqueda` | Qué tan bien salió la mejor combinación y cómo leer ese número. |
| `avisos` | Alertas, por ejemplo si el mejor valor quedó pegado al límite del rango probado. Si está vacío, no hay nada que corregir. |
| `configuracion` | Cuántas pruebas, la semilla y si se usó solo una parte de los animales. |
| `variables_usadas` | Las variables con las que se hizo la búsqueda. |
| `huella_de_dev_features_sha256` | Identifica la versión exacta de `dev_features.csv`. |

El resultado de la búsqueda es una referencia para elegir, no el resultado final del modelo. El resultado final sale del notebook de modelado.

### `trials/<modelo>_<tarea>.csv`: todas las pruebas

Una fila por prueba.

| Columna | Qué significa |
|---|---|
| `prueba` | Número de la prueba, empezando en 1. |
| `resultado` | Qué tan bien salió esa prueba (cuanto más alto, mejor). |
| columnas con el nombre de cada ajuste | El valor que se probó. |
| `duracion_segundos` | Cuánto tardó. |
| `estado` | `completa` si terminó bien. |
| `es_la_mejor` | `sí` en la prueba ganadora. |

Sirve para ver si muchas combinaciones dieron casi lo mismo. En ese caso, el modelo es poco sensible a ese ajuste.

### `figures/<modelo>_<tarea>_*.png`: dos gráficas

Cada gráfica trae debajo una frase que explica cómo leerla.

| Gráfica | Pregunta que responde | Cómo leerla |
|---|---|---|
| `*_historial.png` | ¿La búsqueda fue mejorando? | Cada punto es una prueba y la línea marca la mejor hasta ese momento. Si la línea se aplana pronto, hacer más pruebas ayudaría poco. |
| `*_cortes.png` | ¿Qué valores funcionaron mejor? | Un gráfico por ajuste. Cada punto es una prueba: abajo el valor probado y arriba el resultado. Los puntos más altos son los mejores y los más oscuros se probaron más tarde. Si los mejores quedan en un extremo, conviene ampliar el rango. |

## Cuándo hay que repetir la búsqueda

Siempre que cambie algo de lo que entra al modelo:

- las variables (agregar o quitar),
- `dev_features.csv` (por ejemplo, al volver a ejecutar el notebook 03 con cambios),
- el objetivo.

El notebook de modelado solo comprueba que las variables coincidan con las de la búsqueda, y se detiene con un mensaje si no es así. Un cambio así es un experimento nuevo: se crean las carpetas del experimento 2 y se deja intacto el 1 para poder comparar.

## Cómo se ejecuta

Abrir el notebook del modelo y ejecutar todo. El tiempo depende del modelo: con SVM tarda pocos minutos. Cada notebook usa la misma validación, la misma semilla y el mismo número de pruebas (`N_TRIALS`), para que los modelos sean comparables.
