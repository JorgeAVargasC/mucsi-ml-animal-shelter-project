# mucsi-ml-animal-shelter-project

Proyecto de aprendizaje supervisado de Machine Learning, MUCSI Deusto 2026-27.

**Equipo:** Katherine Lorenzo, Mauro Cornejo, Jorge Vargas

A partir de los datos de ingreso de un animal en el Austin Animal Center, predecir:

| Tarea | Objetivo |
|---|---|
| Clasificación | Resultado de salida (`outcome_type`): Adoption, Transfer, Return to Owner, Euthanasia, Died, Disposal, Rto-Adopt, Missing, Relocate |
| Regresión | Días en el refugio (`time_in_shelter_days`) |

Datos: [Austin Animal Center Shelter Intakes and Outcomes (Kaggle)](https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes), en `data/raw/aac_intakes_outcomes.csv`.

## Estructura

```
data/
  raw/          dataset original
  interim/      particiones dev y test (notebook 01)
  processed/    datos limpios (notebook 02)
notebooks/
  config.py     semilla común
  01-data-collection-split-and-quality.ipynb
  02-eda-and-cleaning.ipynb
  03-feature-engineering-and-transformation.ipynb
```

| Notebook | Entrada | Salida |
|---|---|---|
| 01 Recolección y calidad del dato | `data/raw/` | `data/interim/dev.csv`, `data/interim/test.csv` |
| 02 EDA y limpieza | `data/interim/dev.csv` | `data/processed/` |
| 03 Ingeniería de características y transformación | `data/processed/` | Preprocesador para el modelado |

## Setup

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Seleccionar `.venv` como kernel y ejecutar el notebook 01 para generar `data/interim/`.

## Reglas de trabajo

- En la medida de lo posible, cada persona edita solo su notebook para evitar conflictos.
- `SEED` se importa de `notebooks/config.py` (`from config import SEED`).
- `test.csv` no se explora, solo se usa en la evaluación final.
- `data/interim/` y `data/processed/` no se suben, se generan con los notebooks.
- Antes de cada commit, reiniciar el kernel y ejecutar todo.
