"""EDA reducido para el analista: resumen, distribuciones, correlación y VIF."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .variables import VARIABLES, PALETAS


def tabla_pixeles(stack, claves=None, normalizadas=True) -> pd.DataFrame:
    claves = claves or stack.disponibles
    m = stack.grilla.mascara
    fuente = stack.norm if normalizadas else {k: stack.magnitud(k) for k in claves}
    return pd.DataFrame({k: fuente[k][m] for k in claves})


def resumen(stack) -> pd.DataFrame:
    filas = []
    for k in stack.disponibles:
        v = VARIABLES[k]
        a = stack.magnitud(k)[stack.grilla.mascara]
        a = a[np.isfinite(a)]
        filas.append({"variable": v.nombre, "unidad": v.unidad, "media": a.mean(), "desvío": a.std(),
                      "CV %": 100 * a.std() / abs(a.mean()) if a.mean() else np.nan,
                      "mín": a.min(), "p50": np.median(a), "máx": a.max(),
                      "faltantes %": 100 * (1 - a.size / stack.grilla.mascara.sum())})
    return pd.DataFrame(filas).set_index("variable").round(3)


def vif(df: pd.DataFrame) -> pd.Series:
    """VIF a partir de la inversa de la matriz de correlación."""
    c = df.dropna().corr().to_numpy()
    try:
        inv = np.linalg.inv(c)
        return pd.Series(np.diag(inv), index=df.columns, name="VIF").round(2)
    except np.linalg.LinAlgError:
        return pd.Series(np.inf, index=df.columns, name="VIF")


def pares_correlacionados(corr: pd.DataFrame, umbral=0.85):
    pares = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr.iloc[i, j]) >= umbral:
                pares.append((cols[i], cols[j], round(corr.iloc[i, j], 2)))
    return pares


def graficar(stack, claves=None):
    """Devuelve figuras: mapas normalizados, histogramas y correlación."""
    claves = claves or stack.disponibles
    figs = {}
    n = len(claves)
    ncol = 4
    nfil = int(np.ceil(n / ncol))

    fig, axs = plt.subplots(nfil, ncol, figsize=(4 * ncol, 3.6 * nfil))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, k in zip(np.ravel(axs), claves):
        im = ax.imshow(stack.norm[k], cmap=PALETAS[VARIABLES[k].paleta], vmin=0, vmax=1,
                       extent=stack.grilla.extent)
        ax.set_title(VARIABLES[k].nombre, fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.6, label="valor normalizado (0-1)")
    figs["mapas"] = fig

    df = tabla_pixeles(stack, claves, normalizadas=False)
    fig, axs = plt.subplots(nfil, ncol, figsize=(4 * ncol, 2.8 * nfil))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, k in zip(np.ravel(axs), claves):
        ax.axis("on")
        ax.hist(df[k].dropna(), bins=40, color="#4c72b0")
        ax.set_title(f"{VARIABLES[k].nombre} [{VARIABLES[k].unidad}]", fontsize=9)
    fig.tight_layout()
    figs["histogramas"] = fig

    corr = df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(1 + 0.7 * n, 0.8 + 0.6 * n))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), claves, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), claves, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman")
    ax.set_title("Correlación entre variables")
    fig.tight_layout()
    figs["correlacion"] = fig
    return figs, corr


def ejecutar(stack, claves=None, umbral_corr=0.85):
    claves = claves or stack.disponibles
    figs, corr = graficar(stack, claves)
    v = vif(tabla_pixeles(stack, claves))
    pares = pares_correlacionados(corr, umbral_corr)
    return {"resumen": resumen(stack), "corr": corr, "vif": v, "pares": pares, "figs": figs}
