"""Modelo final del Reto AgroCebada 2026.

Predicción = media de rendimiento del municipio + corrección.
La corrección es una regresión Ridge sobre variables HOMOGENEIZADAS:
  1) cada variable se expresa como anomalía (diferencia respecto al promedio de su municipio, o de su grupo de
     municipios pequeños) -> elimina el efecto de ubicación y deja lo que distingue a la parcela de su zona;
  2) los faltantes se imputan con la mediana;
  3) se estandarizan (z-score) para que todas tengan la misma escala.
Los coeficientes del Ridge sobre variables estandarizadas son los PESOS de las variables.
"""
import re
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

SEED = 42
ALPHAS = np.logspace(1, 4, 20)


# ---------------------------------------------------------------- variables candidatas y anomalías
def columnas_candidatas(t):
    num = t.select_dtypes("number").columns.drop(["rendimiento_t_ha", "lat", "lon"], errors="ignore")
    meses = tuple(f"_m{m:02d}" for m in (2, 3, 4, 5, 6, 7))
    cand = [c for c in num if c.startswith(("pl_", "s2_")) and any(m in c for m in meses)]
    cand += [c for c in ("pl_evi_pico", "pl_ndvi_pico", "pl_evi_auc_mar_jul", "pl_ndvi_auc_mar_jul",
                         "pl_ndvi_std_mar_jul", "pl_evi_pend_abr_jun", "pl_ndvi_pend_abr_jun",
                         "elevacion_m", "pendiente_grados", "area_ha") if c in t.columns]
    cand += [c for c in num if c.startswith(("prec_m0", "tmin_m0", "tmax_m0")) and c[-2:] in ("04", "05", "06", "07")]
    cand += [c for c in ("prec_ciclo", "prec_abr_jun", "tmed_ciclo", "amp_ciclo") if c in t.columns]
    return list(dict.fromkeys(cand))


def agregar_anomalias(t):
    """Devuelve (tabla con columnas *_anom, lista de columnas de anomalía)."""
    t = t.copy()
    n_mun = t.groupby("Municipio").ID_POLIGONO.transform("count")
    t["_ref"] = np.where(n_mun >= 5, t.Municipio, "EST_" + t.Estado)
    cand = columnas_candidatas(t)
    A = t[cand].sub(t.groupby("_ref")[cand].transform("mean")).add_suffix("_anom")
    return pd.concat([t, A], axis=1), list(A.columns)


def media_loo(d):
    """Media del municipio sin contar la propia parcela (evita fuga de información al ajustar)."""
    g = d.groupby("Municipio").rendimiento_t_ha
    s, n = g.transform("sum"), g.transform("count")
    return ((s - d.rendimiento_t_ha) / (n - 1)).where(n > 1, d.groupby("Estado").rendimiento_t_ha.transform("mean"))


class Zonas:
    """Promedios de referencia por zona para convertir valores de una parcela nueva en anomalías."""
    def __init__(self, t):
        self.cand = columnas_candidatas(t)
        n_mun = t.groupby("Municipio").ID_POLIGONO.transform("count")
        ref = np.where(n_mun >= 5, t.Municipio, "EST_" + t.Estado)
        self.medias = t[self.cand].groupby(ref).mean()
        self.global_ = t[self.cand].mean()
        self.ok_mun = set(t.Municipio[n_mun >= 5])
        self.mun_estado = t.drop_duplicates("Municipio").set_index("Municipio").Estado.to_dict()

    def clave(self, mun, est):
        if mun in self.ok_mun: return mun
        if f"EST_{est}" in self.medias.index: return f"EST_{est}"
        return None

    def referencia(self, mun, est):
        k = self.clave(mun, est)
        return self.global_ if k is None else self.medias.loc[k]

    def anomalias(self, df):
        """Para cada fila: valor - promedio de su zona. Si falta el valor de una variable, la anomalía es 0."""
        out = pd.DataFrame(0.0, index=df.index, columns=[c + "_anom" for c in self.cand])
        for i, r in df.iterrows():
            ref = self.referencia(r.get("Municipio"), r.get("Estado"))
            for c in self.cand:
                if c in df.columns and pd.notna(r[c]):
                    out.at[i, c + "_anom"] = float(r[c]) - float(ref[c])
        return out


# ---------------------------------------------------------------- modelo
class ModeloAgroCebada:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, d):
        self.mun = d.groupby("Municipio").rendimiento_t_ha.mean()
        self.est = d.groupby("Estado").rendimiento_t_ha.mean()
        self.glob = d.rendimiento_t_ha.mean()
        if not self.cols:                       # sin variables: queda solo la media del municipio
            self.pipe = None
            return self
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=ALPHAS))
        self.pipe.fit(d[self.cols], (d.rendimiento_t_ha - media_loo(d)).values)
        return self

    @property
    def alpha(self):
        return float(self.pipe.named_steps["ridgecv"].alpha_)

    def base(self, df):
        return np.array([self.mun.get(m, self.est.get(e, self.glob)) for m, e in zip(df.Municipio, df.Estado)])

    def correccion(self, df):
        return np.zeros(len(df)) if self.pipe is None else self.pipe.predict(df[self.cols])

    def predict(self, df):
        return self.base(df) + self.correccion(df)

    def pesos(self):
        """Peso de cada variable = coeficiente Ridge sobre la variable estandarizada (y su % del total)."""
        coef = self.pipe.named_steps["ridgecv"].coef_
        d = pd.DataFrame({"variable": [c.replace("_anom", "") for c in self.cols], "coeficiente": coef})
        d["peso_pct"] = d.coeficiente.abs() / d.coeficiente.abs().sum() * 100
        d["etiqueta"] = d.variable.map(etiqueta)
        d["grupo"] = d.variable.map(grupo_variable)
        d["familia_indice"] = d.variable.map(familia_indice)
        return d.sort_values("peso_pct", ascending=False).reset_index(drop=True)

    def contribuciones(self, df):
        """Aporte de cada variable (coeficiente x valor estandarizado) a la corrección de una parcela."""
        z = self.pipe.named_steps["standardscaler"].transform(self.pipe.named_steps["simpleimputer"].transform(df[self.cols]))
        coef = self.pipe.named_steps["ridgecv"].coef_
        return pd.Series(z[0] * coef, index=[etiqueta(c) for c in self.cols]).sort_values(key=abs, ascending=False)


def predecir_nuevas(modelo, zonas, df, q10, q90):
    """Predice parcelas nuevas a partir de valores crudos (Municipio es obligatorio; el resto es opcional)."""
    df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True).copy()
    if "Estado" not in df.columns: df["Estado"] = np.nan
    df["Estado"] = df.Estado.fillna(df.Municipio.map(zonas.mun_estado))
    X = pd.concat([df[["Municipio", "Estado"]], zonas.anomalias(df)], axis=1)
    base, cor = modelo.base(X), modelo.correccion(X)
    out = pd.DataFrame({"Municipio": df.Municipio, "Estado": df.Estado, "media_municipio": base, "correccion": cor,
                        "rendimiento_predicho": base + cor})
    out["p10"], out["p90"] = out.rendimiento_predicho + q10, out.rendimiento_predicho + q90
    if "area_ha" in df.columns:
        out["produccion_esperada_t"] = out.rendimiento_predicho * df.area_ha.astype(float)
    return out, X


# ---------------------------------------------------------------- etiquetas y grupos
_MESES = {"01": "enero", "02": "febrero", "03": "marzo", "04": "abril", "05": "mayo", "06": "junio",
          "07": "julio", "08": "agosto", "09": "septiembre", "10": "octubre"}
_IND = {"ndvi": "NDVI", "evi": "EVI", "ndwi": "NDWI", "msi": "MSI", "crc": "CRC", "dswi2": "DSWI2"}


def etiqueta(col):
    c = col.replace("_anom", "")
    m = re.fullmatch(r"(pl|s2)_([a-z0-9]+)_m(\d\d)", c)
    if m:
        s = "Planet" if m.group(1) == "pl" else "Sentinel-2"
        return f"{s} {_IND.get(m.group(2), m.group(2))} {_MESES[m.group(3)]}"
    m = re.fullmatch(r"(prec|tmin|tmax)_m(\d\d)", c)
    if m:
        n = {"prec": "Precipitación", "tmin": "Tmin", "tmax": "Tmax"}[m.group(1)]
        return f"{n} {_MESES[m.group(2)]}"
    otros = {"pl_evi_pico": "Planet EVI pico", "pl_ndvi_pico": "Planet NDVI pico",
             "pl_evi_auc_mar_jul": "Planet EVI área bajo la curva (mar-jul)",
             "pl_ndvi_auc_mar_jul": "Planet NDVI área bajo la curva (mar-jul)",
             "pl_ndvi_std_mar_jul": "Planet NDVI heterogeneidad (mar-jul)",
             "pl_evi_pend_abr_jun": "Planet EVI cambio abril→junio",
             "pl_ndvi_pend_abr_jun": "Planet NDVI cambio abril→junio",
             "elevacion_m": "Elevación (m)", "pendiente_grados": "Pendiente (°)", "area_ha": "Área (ha)",
             "prec_ciclo": "Precipitación del ciclo", "prec_abr_jun": "Precipitación abr-jun",
             "tmed_ciclo": "Temperatura media del ciclo", "amp_ciclo": "Amplitud térmica del ciclo"}
    return otros.get(c, c)


def grupo_variable(col):
    c = col.replace("_anom", "")
    if c.startswith("pl_"): return "Satelital: Planet"
    if c.startswith("s2_"): return "Satelital: Sentinel-2"
    if c.startswith(("prec", "tmin", "tmax", "tmed", "amp")): return "Clima"
    return "Topografía y parcela"


def familia_indice(col):
    m = re.match(r"^(pl|s2)_(ndvi|evi|ndwi|msi|crc|dswi2)", col.replace("_anom", ""))
    if not m: return None
    return f"{'Planet' if m.group(1) == 'pl' else 'Sentinel-2'} {_IND[m.group(2)]}"
