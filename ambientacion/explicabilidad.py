"""¿Qué variables explican la delimitación? Random Forest + SHAP + Kruskal-Wallis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .variables import VARIABLES


def explicar(datos, etiquetas, colores, nombres, cfg, muestra=20000, muestra_shap=1500):
    """`etiquetas`: id de ambiente (1..k) por píxel, alineado con datos.X."""
    rng = np.random.default_rng(cfg.semilla)
    n = len(etiquetas)
    idx = rng.choice(n, min(muestra, n), replace=False)
    X = pd.DataFrame(datos.X[idx], columns=datos.claves)
    y = etiquetas[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=cfg.semilla, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=cfg.semilla)
    rf.fit(Xtr, ytr)
    exactitud = rf.score(Xte, yte)
    perm = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=cfg.semilla, n_jobs=-1)

    tabla = pd.DataFrame({
        "variable": [VARIABLES[c].nombre for c in datos.claves],
        "importancia_permutacion": perm.importances_mean,
    }, index=datos.claves)

    # Kruskal-Wallis y tamaño de efecto (épsilon²)
    k = len(np.unique(y))
    for c in datos.claves:
        grupos = [X.loc[y == g, c] for g in np.unique(y)]
        h, p = kruskal(*grupos)
        tabla.loc[c, "H_kruskal"] = h
        tabla.loc[c, "epsilon2"] = (h - k + 1) / (len(y) - k)
        tabla.loc[c, "p_valor"] = p

    # SHAP
    shap_ambiente = None
    try:
        import shap
        Xs = Xte.iloc[: min(muestra_shap, len(Xte))]
        sv = shap.TreeExplainer(rf).shap_values(Xs)
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        # sv: (n, variables, clases)
        shap_ambiente = pd.DataFrame(np.abs(sv).mean(axis=0), index=datos.claves, columns=rf.classes_)
        tabla["shap_medio"] = shap_ambiente.sum(axis=1)
    except Exception as e:  # noqa: BLE001
        print(f"SHAP no disponible ({e}); se informa importancia por permutación.")

    orden = "shap_medio" if "shap_medio" in tabla else "importancia_permutacion"
    tabla = tabla.sort_values(orden, ascending=False)
    fig = _grafico(tabla, shap_ambiente, colores, nombres, orden)
    return {"tabla": tabla.round(4), "shap": shap_ambiente, "exactitud_rf": exactitud, "fig": fig}


def _grafico(tabla, shap_amb, colores, nombres, orden):
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(tabla) + 1.5))
    etiquetas = tabla["variable"].tolist()[::-1]
    if shap_amb is not None:
        base = np.zeros(len(tabla))
        for amb in shap_amb.columns:
            vals = shap_amb.loc[tabla.index[::-1], amb].to_numpy()
            ax.barh(etiquetas, vals, left=base, color=colores[amb], label=nombres[amb],
                    edgecolor="white", lw=0.5)
            base += vals
        ax.set_xlabel("|SHAP| medio (contribución a la asignación de cada ambiente)")
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.barh(etiquetas, tabla[orden].to_numpy()[::-1], color="#4c72b0")
        ax.set_xlabel("Importancia por permutación")
    ax.set_title("Variables que explican la delimitación de ambientes")
    fig.tight_layout()
    return fig
