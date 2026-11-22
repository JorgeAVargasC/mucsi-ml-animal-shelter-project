"""Ingeniería de características (notebook 03).

Solo usa información conocida al ingreso del animal. Las funciones no aprenden nada
de los datos; lo que sí se aprende (categorías frecuentes, escalado) está dentro del
preprocesador, que se ajusta únicamente con los datos de entrenamiento.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from config import SEED

ADOPTED = ["Adoption", "Rto-Adopt"]
MAIN_OUTCOMES = ["Transfer", "Return to Owner", "Euthanasia"]
SERIOUS = ["Injured", "Sick", "Nursing", "Aged"]
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}

# Columnas que usa el preprocesador
CAT_LOW = ["animal_type", "intake_type", "intake_condition", "sex", "weekday", "season"]
CAT_HIGH = ["breed_main", "color_main", "city"]
BINARY = ["is_neutered", "is_mix", "is_crossbreed", "is_bicolor", "is_tabby", "has_location_detail",
          "is_baby", "condition_serious"]
NUMERIC = ["age_days", "prior_stays", "hour", "month", "t_months"]
FEATURES = CAT_LOW + CAT_HIGH + BINARY + NUMERIC

# Experimento 2: edad en logaritmo. No está en dev_features.csv, se calcula con add_age_log()
NUMERIC_LOG = ["age_log"]


def add_age_log(df):
    """Devuelve una copia de df con age_log = log(1 + age_days).

    La edad tiene cola larga: pocos animales muy viejos dominan las distancias. El logaritmo
    acerca esos valores, y la diferencia entre 1 y 3 meses pesa como la de 5 y 15 años.
    """
    out = df.copy()
    out["age_log"] = np.log1p(out["age_days"])
    return out


def add_features(df):
    """Devuelve una copia de df con las columnas nuevas. No usa columnas de salida.

    prior_stays cuenta los ingresos anteriores del mismo animal (por fecha), así que todos los
    ingresos de un animal deben estar en el mismo conjunto. Con una partición por animal se cumple.
    """
    out = df.copy()
    when = pd.to_datetime(out["intake_datetime"])

    # Raza: mestizo, cruce de dos razas y raza principal sin "Mix"
    out["is_mix"] = out["breed"].str.contains("Mix").astype(int)
    out["is_crossbreed"] = out["breed"].str.contains("/").astype(int)
    out["breed_main"] = out["breed"].str.split("/").str[0].str.replace(r"\s+Mix$", "", regex=True)

    # Color: primer color, si tiene dos y si es atigrado
    out["color_main"] = out["color"].str.split("/").str[0]
    out["is_bicolor"] = out["color"].str.contains("/").astype(int)
    out["is_tabby"] = out["color"].str.contains("Tabby").astype(int)

    # Lugar: "9700 Colfax Dr in Austin (TX)" -> ciudad "Austin"; "Austin (TX)" no trae dirección
    out["city"] = (out["found_location"].str.replace(r"\s*\([A-Za-z]{2}\)\s*$", "", regex=True)
                   .str.split(" in ").str[-1].str.strip())
    out["has_location_detail"] = out["found_location"].str.contains(" in ").astype(int)

    # Sexo y esterilización por separado. "Unknown" no trae ninguna de las dos:
    # queda sex = "Unknown" e is_neutered = 0
    sex = out["sex_upon_intake"]
    out["sex"] = np.select([sex.str.contains("Female|Spayed"), sex.str.contains("Male")],
                           ["Female", "Male"], default="Unknown")
    out["is_neutered"] = sex.str.contains("Neutered|Spayed").astype(int)

    # Edad, condición e historial
    out["age_days"] = out["age_upon_intake_(days)"]
    out["is_baby"] = (out["age_days"] < 365).astype(int)
    out["condition_serious"] = out["intake_condition"].isin(SERIOUS).astype(int)
    # intake_number no sigue el orden por fecha, así que se cuenta con intake_datetime
    out["prior_stays"] = (when.groupby(out["animal_id_intake"]).rank(method="first") - 1).astype(int)

    # Tiempo
    out["hour"] = when.dt.hour
    out["month"] = when.dt.month
    out["weekday"] = when.dt.day_name()
    out["season"] = when.dt.month.map(SEASONS)
    out["t_months"] = (when.dt.year - 2013) * 12 + when.dt.month
    return out


def make_targets(df):
    """Objetivos. y_outcome: 5 clases (Rto-Adopt cuenta como Adoption).
    y_log_days solo existe para los adoptados (NaN en el resto)."""
    adopted = df["outcome_type"].isin(ADOPTED)
    multi = df["outcome_type"].where(df["outcome_type"].isin(MAIN_OUTCOMES), "Other")
    return pd.DataFrame({
        "y_outcome": multi.mask(adopted, "Adoption"),
        "y_log_days": np.log1p(df["time_in_shelter_days"]).where(adopted),
    }, index=df.index)


# def build_preprocessor(cat_low=CAT_LOW, cat_high=CAT_HIGH, binary=BINARY, numeric=NUMERIC,
#                        min_frequency=0.01, columns=None):
#     """Preprocesador sin ajustar. Las categorías con menos del 1 % se agrupan en 'infrequent'.

#     Si se pasa `columns`, solo se usan esas variables.
#     """
#     if columns is not None:
#         cat_low, cat_high, binary, numeric = ([c for c in group if c in columns]
#                                               for group in (cat_low, cat_high, binary, numeric))
#     low = OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)
#     high = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
#                          sparse_output=False)
#     return ColumnTransformer([
#         ("cat_low", low, list(cat_low)),
#         ("cat_high", high, list(cat_high)),
#         ("num", StandardScaler(), list(numeric)),
#         ("binary", "passthrough", list(binary)),
#     ])

def build_preprocessor(cat_low=CAT_LOW, cat_high=CAT_HIGH, binary=BINARY, numeric=NUMERIC,
                       min_frequency=0.01, columns=None, numeric_scaler=None, cat_high_encoding="onehot"):
    """Preprocesador sin ajustar. Las categorías con menos del 1 % se agrupan en 'infrequent'.

    Si se pasa `columns`, solo se usan esas variables.

    cat_high_encoding="target" (experimento 2 de los árboles): raza, color y ciudad se cambian por el
    promedio del objetivo en cada categoría, en lugar de one-hot. Así no se pierde ninguna categoría.
    TargetEncoder lo calcula con validación cruzada interna al entrenar, para no filtrar el objetivo.
    En clasificación crea una columna por clase y variable (la proporción de cada salida).
    """
    if columns is not None:
        cat_low, cat_high, binary, numeric = ([c for c in group if c in columns]
                                              for group in (cat_low, cat_high, binary, numeric))

    low = OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False)
    if cat_high_encoding == "target":
        high = TargetEncoder(target_type="auto", cv=KFold(n_splits=5, shuffle=True, random_state=SEED))
    else:
        high = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                             sparse_output=False)

    if numeric_scaler is None:
        numeric_scaler = StandardScaler()

    return ColumnTransformer([
        ("cat_low", low, list(cat_low)),
        ("cat_high", high, list(cat_high)),
        ("num", numeric_scaler, list(numeric)),
        ("binary", "passthrough", list(binary)),
    ])