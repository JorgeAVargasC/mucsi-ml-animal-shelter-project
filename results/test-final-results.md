# Resultados finales en test

Resultados del notebook `05-final-training-and-test.ipynb`: el modelo final, entrenado con todo el conjunto de desarrollo y evaluado una sola vez con `test`.

## Modelo

Random forest con las variables y los hiperparámetros del experimento 1, en clasificación y en regresión. La elección y sus motivos están en [`cross-validation-results.md`](cross-validation-results.md).

| | Clasificación | Regresión |
|---|---|---|
| Objetivo | `outcome_type` (5 clases) | Días hasta la adopción (en log) |
| Filas de entrenamiento (dev) | 63.681 | 27.073 (solo adoptados) |
| Filas de evaluación (test) | 15.946 | 6.689 (solo adoptados) |
| `n_estimators` | 200 | 200 |
| `max_depth` | 38 | 20 |
| `min_samples_leaf` | 1 | 2 |
| `max_features` | 0,298 | 0,387 |
| `class_weight` | ninguno | no aplica |

Los hiperparámetros salen de `04-hyperparameter-optimization-experiment-1/best_params/` (`rf_clf.json` y `rf_reg.json`). No se repitió la búsqueda.

## Resultados en test

| Métrica | Modelo | Línea base |
|---|---|---|
| macro-F1 | **0,578** | 0,118 (siempre la clase más frecuente) |
| R² (log) | **0,412** | −0,051 (siempre la mediana) |
| MAE (días) | **18,393** | 24,130 (siempre la mediana) |

La exactitud (*accuracy*) en clasificación es 0,67.

## Test frente a validación cruzada

| Métrica | Validación cruzada (dev) | Test | Diferencia |
|---|---|---|---|
| macro-F1 | 0,576 | 0,578 | +0,002 |
| R² (log) | 0,413 | 0,412 | −0,001 |
| MAE (días) | 18,556 | 18,393 | −0,163 |

Los resultados en test son prácticamente iguales a los de la validación cruzada. No hay señal de que el modelo se haya ajustado de más a dev.

## Clasificación por clase

| Clase | Precisión | Recall | F1 | Casos en test |
|---|---|---|---|---|
| Adoption | 0,65 | 0,79 | 0,72 | 6.689 |
| Euthanasia | 0,79 | 0,65 | 0,71 | 1.252 |
| Return to Owner | 0,65 | 0,64 | 0,64 | 3.027 |
| Transfer | 0,71 | 0,55 | 0,62 | 4.760 |
| Other | 0,44 | 0,13 | 0,20 | 218 |

- `Adoption` y `Euthanasia` son las clases mejor resueltas (F1 de 0,72 y 0,71).
- `Transfer` tiene recall bajo (0,55): casi la mitad de las transferencias se predice como otra salida.
- `Other` es la clase más difícil: solo se detecta el 13 % de los casos. Es la más pequeña (218 casos, el 1,4 % de test) y agrupa salidas muy distintas entre sí (`Died`, `Disposal`, `Missing`, `Relocate`).

## Regresión

El R² se calcula sobre el logaritmo de los días y el error se reporta en días. El modelo reduce el error de la línea base de 24,1 a 18,4 días y explica el 41 % de la variación en log.

## Conclusiones

- El modelo generaliza bien: en test rinde igual que en validación cruzada. Usar las variables del experimento 1 también en regresión no tuvo un costo apreciable.
- En clasificación supera con mucha ventaja a la línea base (macro-F1 de 0,578 frente a 0,118) y acierta unas 2 de cada 3 salidas. Aun así, es una tarea difícil con la información disponible al ingreso.
- Las adopciones y las eutanasias son lo que mejor se distingue. `Return to Owner` y `Transfer` quedan en un punto intermedio (F1 de 0,64 y 0,62), y `Other` casi no se detecta, por lo que es el principal punto débil.
- En regresión, el modelo sirve como orientación y no como predicción exacta: el error medio es de unos 18 días, seis menos que la línea base, y explica algo menos de la mitad de la variación en log.

## Limitaciones

- Los hiperparámetros del random forest se buscaron con el 20 % de los animales.
