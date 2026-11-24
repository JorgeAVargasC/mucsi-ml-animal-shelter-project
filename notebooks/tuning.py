"""Búsqueda de hiperparámetros con Optuna (notebooks 04).

tune() busca los mejores hiperparámetros de un modelo y guarda tres tipos de archivos, todas con
textos explicados en español:
  best_params/<nombre>.json   los mejores valores encontrados
  trials/<nombre>.csv         todas las pruebas que se hicieron
  figures/<nombre>_*.png      dos gráficas con su interpretación

load_best() lee el JSON en el notebook de modelado y comprueba que las variables y los
datos son los mismos con los que se hizo la búsqueda.
"""
import datetime
import hashlib
import json
import platform
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import sklearn
from optuna.visualization import matplotlib as optuna_plots
from sklearn.model_selection import GroupKFold, cross_val_score

from config import SEED

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "dev_features.csv"

# nombre en pantalla y explicación sencilla de cada métrica
METRICS = {
    "f1_macro": ("macro-F1",
                 "Promedio del acierto en cada clase, de 0 a 1. Cuanto más alto, mejor. "
                 "Cuenta igual a las clases grandes y a las pequeñas."),
    "r2": ("R² (en log)",
           "Qué parte de la variación de los días logra explicar el modelo. 1 es perfecto "
           "y 0 es lo mismo que predecir siempre el promedio."),
}

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _progress(n_trials):
    def callback(study, trial):
        value = "falló" if trial.value is None else f"{trial.value:.3f}"
        print(f"Prueba {trial.number + 1}/{n_trials}: resultado {value} | mejor hasta ahora {study.best_value:.3f}")
    return callback


def _edge_notes(study):
    """Avisa si el mejor valor de un ajuste numérico quedó pegado al límite de su rango."""
    notes = []
    for name, dist in study.best_trial.distributions.items():
        if not hasattr(dist, "low"):
            continue
        best, low, high = study.best_params[name], dist.low, dist.high
        if getattr(dist, "log", False):
            pos = (np.log(best) - np.log(low)) / (np.log(high) - np.log(low))
        else:
            pos = (best - low) / (high - low)
        if pos <= 0.05:
            notes.append(f"El mejor valor de {name} ({best:.3g}) está pegado al límite inferior del rango ({low:g}). "
                         "Conviene ampliar el rango hacia abajo y repetir la búsqueda.")
        elif pos >= 0.95:
            notes.append(f"El mejor valor de {name} ({best:.3g}) está pegado al límite superior del rango ({high:g}). "
                         "Conviene ampliar el rango hacia arriba y repetir la búsqueda.")
    return notes


def _caption(fig, text):
    width = max(60, int(fig.get_figwidth() * 13))
    fig.set_layout_engine("none")
    fig.subplots_adjust(bottom=0.30)
    fig.text(0.02, 0.01, textwrap.fill("Cómo leerla: " + text, width), ha="left", va="bottom", fontsize=9)


def _from_one(axis):
    """Optuna numera las pruebas desde 0; aquí se muestran desde 1, como en el CSV."""
    axis.set_major_locator(plt.MaxNLocator(integer=True))
    axis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{round(value) + 1}"))


def _save(fig, path):
    fig.savefig(path, dpi=130)


def _make_figures(study, name, metric_label, fig_dir):
    figures = {}

    # 1) historial
    ax = optuna_plots.plot_optimization_history(study)
    ax.figure.set_size_inches(8, 5)
    ax.set_title("¿La búsqueda fue mejorando?")
    ax.set_xlabel("Número de prueba")
    ax.set_ylabel(metric_label)
    _from_one(ax.xaxis)
    names = {"Objective Value": "Cada prueba", "Best Value": "Mejor hasta ese momento"}
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [names.get(label, label) for label in labels])
    _caption(ax.figure, "cada punto es una prueba y la línea marca la mejor hasta ese momento. "
                        "Si la línea se aplana pronto, hacer más pruebas ayudaría poco.")
    _save(ax.figure, fig_dir / f"{name}_historial.png")
    figures["historial"] = ax.figure

    # 2) cortes por ajuste
    axes = np.atleast_1d(optuna_plots.plot_slice(study)).ravel()
    fig = axes[0].figure
    fig.set_size_inches(4.5 * len(axes), 5)
    axes[0].set_ylabel(metric_label)
    for ax in axes[1:]:
        ax.set_ylabel("")
    for ax in axes:
        if ax.get_xscale() == "log":  # sin etiquetas intermedias, que se amontonan
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
    fig.suptitle("¿Qué valores funcionaron mejor?")
    colorbar = fig.axes[-1]
    colorbar.set_ylabel("Orden de la prueba")
    _from_one(colorbar.yaxis)
    _caption(fig, "cada punto es una prueba: abajo el valor probado de ese ajuste y arriba el resultado obtenido. "
                  "Los puntos más altos son los mejores y los más oscuros se probaron más tarde. "
                  "Si los mejores quedan en un extremo, conviene ampliar el rango.")
    fig.subplots_adjust(left=0.07, right=0.88, top=0.90)
    colorbar.set_position([0.91, 0.36, 0.015, 0.52])
    _save(fig, fig_dir / f"{name}_cortes.png")
    figures["cortes"] = fig
    return figures


def tune(name, make_pipeline, space, X, y, groups, scoring, features, *,
         model_label, task_label, params_help=None, n_trials=20, subsample=None,
         out_dir=".", n_jobs=5, seed=SEED):
    """Busca los mejores hiperparámetros con Optuna y guarda el resultado.

    name           nombre de los archivos, por ejemplo "svm_clf"
    make_pipeline  función que recibe un dict de parámetros y devuelve el pipeline sin ajustar
    space          función que recibe un `trial` de Optuna y devuelve el dict de parámetros
    scoring        métrica de sklearn: "f1_macro" o "r2"
    features       lista de variables que usa el modelo (se guarda para comprobarla después)
    subsample      fracción de animales a usar para que la búsqueda sea rápida (None = todos)
    params_help    dict opcional {ajuste: explicación sencilla} que se guarda en el JSON
    """
    metric_label, metric_help = METRICS.get(scoring, (scoring, ""))
    out_dir = Path(out_dir)
    for folder in ("best_params", "trials", "figures"):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)

    if subsample:
        rng = np.random.RandomState(seed)
        animals = rng.choice(groups.unique(), size=int(subsample * groups.nunique()), replace=False)
        keep = groups.isin(animals)
        X, y, groups = X[keep], y[keep], groups[keep]
    print(f"Búsqueda de {model_label} ({task_label}): {n_trials} pruebas con {len(X):,} filas y {groups.nunique():,} animales.")

    cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)

    def objective(trial):
        pipeline = make_pipeline(space(trial))
        return cross_val_score(pipeline, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=n_jobs).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, callbacks=[_progress(n_trials)])

    notes = _edge_notes(study)
    print(f"\nMejores ajustes: {study.best_params}")
    print(f"Resultado de la búsqueda: {metric_label} = {study.best_value:.3f}")
    for note in notes or ["Los mejores valores quedaron dentro de los rangos probados."]:
        print("-", note)

    # JSON con los mejores parámetros
    subsample_text = (f"Se midió con el {subsample:.0%} de los animales para que fuera rápido, por eso es una "
                      "referencia para elegir y no el resultado final del modelo.") if subsample else \
                     "Se midió con todos los datos de desarrollo."
    summary = {
        "que_es": "Los mejores ajustes (hiperparámetros) que encontró Optuna para este modelo. "
                  "El notebook de modelado los usa tal cual.",
        "modelo": model_label,
        "tarea": task_label,
        "mejores_parametros": study.best_params,
        "significado_de_cada_ajuste": params_help or {},
        "resultado_de_la_busqueda": {
            "metrica": metric_label,
            "valor": round(study.best_value, 4),
            "como_leerlo": f"{metric_help} Es el promedio de 5 validaciones en las que los animales de "
                           f"prueba no aparecen en el entrenamiento. {subsample_text}",
        },
        "avisos": notes,
        "configuracion": {
            "pruebas": n_trials,
            "semilla": seed,
            "validacion": "5 grupos, separando los ingresos de cada animal",
            "fraccion_de_animales_usada": subsample,
        },
        "variables_usadas": list(features),
        "huella_de_dev_features_sha256": _sha256(DATA_PATH),
        "huella_explicacion": "Identifica la versión exacta de dev_features.csv. Si el archivo cambia, "
                              "hay que repetir la búsqueda.",
        "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
        "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "optuna": optuna.__version__},
    }
    (out_dir / "best_params" / f"{name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV con todas las pruebas
    trials = study.trials_dataframe(attrs=("number", "value", "params", "duration", "state"))
    trials.columns = [c.replace("params_", "") for c in trials.columns]
    trials["number"] += 1
    trials["duration"] = trials["duration"].dt.total_seconds().round(1)
    trials["state"] = trials["state"].map({"COMPLETE": "completa", "FAIL": "falló", "PRUNED": "cortada"})
    trials["es_la_mejor"] = np.where(trials["number"] == study.best_trial.number + 1, "sí", "")
    trials = trials.rename(columns={"number": "prueba", "value": "resultado",
                                    "duration": "duracion_segundos", "state": "estado"})
    trials.to_csv(out_dir / "trials" / f"{name}.csv", index=False, float_format="%.6g")

    _make_figures(study, name, metric_label, out_dir / "figures")
    print(f"\nGuardado en {out_dir.resolve()}: best_params/{name}.json, trials/{name}.csv y 2 gráficas en figures/")
    return study


def load_best(name, features, folder="../04-hyperparameter-optimization-experiment-1"):
    """Lee los mejores ajustes guardados y comprueba que se buscaron con las mismas variables."""
    path = Path(folder) / "best_params" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero el notebook de optimización de este modelo.")
    saved = json.loads(path.read_text(encoding="utf-8"))

    if sorted(saved["variables_usadas"]) != sorted(features):
        raise ValueError("Las variables de este notebook no son las que se usaron en la búsqueda de ajustes. "
                         "Repite la búsqueda con estas variables.")

    result = saved["resultado_de_la_busqueda"]
    print(f"Ajustes cargados de {path.name}: {saved['mejores_parametros']}")
    print(f"(en la búsqueda: {result['metrica']} = {result['valor']})")
    return saved["mejores_parametros"]
