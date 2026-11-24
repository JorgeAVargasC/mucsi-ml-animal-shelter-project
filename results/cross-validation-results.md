# Resumen de resultados de validación cruzada

Se analizaron los resultados de los notebooks de la seccion 04, la validación cruzada de varios modelos.

## Líneas base

| Tarea | Línea base | Resultado |
|---|---|---|
| Clasificación | Predecir siempre la clase más frecuente | macro-F1 0,119 |
| Regresión | Predecir siempre la mediana | R² (log) −0,058 · MAE 24,312 días |

## Clasificación · macro-F1 (más alto es mejor)

| Modelo | Experimento 1 | Experimento 2 |
|---|---|---|
| **Random forest** | **0,576** | **0,566** |
| Árbol de decisión | 0,547 | 0,540 |
| KNN | 0,534 | 0,529 |
| SVM | 0,501 | 0,513 |
| Regresión logística | 0,481 | 0,481 |
| Naive Bayes | 0,443 | 0,444 |

## Regresión · R² en log (más alto es mejor)

| Modelo | Experimento 1 | Experimento 2 |
|---|---|---|
| **Random forest** | **0,413** | **0,431** |
| Árbol de decisión | 0,347 | 0,384 |
| SVM | 0,214 | 0,345 |
| KNN | 0,236 | 0,290 |
| Regresión lineal | 0,222 | 0,219 |

Naive Bayes solo clasifica.

## Regresión · MAE en días (más bajo es mejor)

| Modelo | Experimento 1 | Experimento 2 |
|---|---|---|
| **Random forest** | **18,556** | **18,615** |
| Árbol de decisión | 19,492 | 19,247 |
| SVM | 21,373 | 19,626 |
| KNN | 20,975 | 20,471 |
| Regresión lineal | 22,229 | 22,053 |

## Interpretación de resultados

- El random forest tiene el mejor valor de todos los experimentos, en clasificación y en regresión.
- Ventaja del random forest sobre el segundo modelo (el árbol de decisión):
  - macro-F1: +0,029 en el experimento 1 y +0,026 en el 2.
  - R²: +0,066 en el experimento 1 y +0,047 en el 2.
- Entre experimentos, el random forest no gana siempre en el mismo:
  - El experimento 1 gana en macro-F1 (0,576 contra 0,566) y en MAE (18,556 contra 18,615).
  - El experimento 2 gana en R² (0,431 contra 0,413).
- El experimento 2 mejora sobre todo a SVM y KNN en regresión. El R² de SVM sube de 0,214 a 0,345.
- La clase `Other` es la más difícil. En el random forest del experimento 1, su F1 es 0,22.

## Modelo elegido

Se elige el random forest con las variables (inputs) y los hiperparámetros del experimento 1, en clasificación y en regresión.

- En clasificación, el experimento 1 es el mejor (macro-F1 0,576 contra 0,566).
- En regresión, el experimento 2 tiene mejor R² (0,431 contra 0,413), pero la mejora es pequeña y el MAE es casi igual (18,556 contra 18,615 días, ligeramente mejor en el 1). Por practicidad, se usan los mismos inputs en las dos tareas.

Limitación conocida: los hiperparámetros del random forest se buscaron con el 20 % de los animales.
