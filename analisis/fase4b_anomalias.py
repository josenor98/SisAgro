"""
Reto AgroCebada 2026 - CRISP-DM Fase 4 (ronda 2): predicción = media del municipio + corrección por anomalías.
Cada variable se expresa como su diferencia respecto al promedio de su municipio (sin usar el rendimiento,
así que se calcula con las 197 parcelas). El modelo solo aprende la desviación de la parcela respecto a su municipio.
Uso: python fase4b_anomalias.py tabla_variables_agrocebada.csv
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore"); SEED = 42
t = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "tabla_variables_agrocebada.csv")

# ---- anomalías respecto al municipio (si el municipio tiene <5 parcelas, respecto al estado)
n_mun = t.groupby("Municipio").ID_POLIGONO.transform("count")
ref = np.where(n_mun >= 5, t.Municipio, "EST_" + t.Estado)
t["_ref"] = ref
num = t.select_dtypes("number").columns.drop(["rendimiento_t_ha", "lat", "lon"])
cand = [c for c in num if c.startswith(("pl_", "s2_")) and any(f"_m{m:02d}" in c for m in (2, 3, 4, 5, 6, 7))
        or c in ("pl_evi_pico", "pl_ndvi_pico", "pl_evi_auc_mar_jul", "pl_ndvi_auc_mar_jul", "pl_ndvi_std_mar_jul",
                 "pl_evi_pend_abr_jun", "pl_ndvi_pend_abr_jun", "elevacion_m", "pendiente_grados", "area_ha")]
cand += [c for c in num if c.startswith(("prec_m0", "tmin_m0", "tmax_m0")) and c[-2:] in ("04", "05", "06", "07")]
cand += ["prec_ciclo", "prec_abr_jun", "tmed_ciclo", "amp_ciclo"]
cand = list(dict.fromkeys(cand))
A = t[cand].sub(t.groupby("_ref")[cand].transform("mean")).add_suffix("_anom")
t = pd.concat([t, A], axis=1)
anom = list(A.columns)
sat_a = [c for c in anom if c.startswith(("pl_", "s2_"))]
cli_a = [c for c in anom if c.startswith(("prec", "tmin", "tmax", "tmed", "amp"))]
top_a = [c for c in anom if c.startswith(("elev", "pend", "area"))]

tr = t[t.conjunto == "ENTRENAMIENTO"].reset_index(drop=True); y = tr.rendimiento_t_ha.values
tr["pixel"] = (np.floor((tr.lon + 99.94999880716205) / 0.05).astype(int).astype(str) + "_" +
               np.floor((21.39999957382679 - tr.lat) / 0.05).astype(int).astype(str))

def loo_mean(d):
    g = d.groupby("Municipio").rendimiento_t_ha; s, n = g.transform("sum"), g.transform("count")
    return ((s - d.rendimiento_t_ha) / (n - 1)).where(n > 1, d.groupby("Estado").rendimiento_t_ha.transform("mean"))

def cv(splits, mk, cols):
    oof, k = np.zeros(len(y)), np.zeros(len(y))
    for a, b in splits:
        d = tr.iloc[a]; mm = d.groupby("Municipio").rendimiento_t_ha.mean()
        ee = d.groupby("Estado").rendimiento_t_ha.mean(); gm = d.rendimiento_t_ha.mean()
        base = lambda df: np.array([mm.get(m, ee.get(e, gm)) for m, e in zip(df.Municipio, df.Estado)])
        if mk is None: oof[b] += base(tr.iloc[b])
        else:
            m = mk(); m.fit(d[cols], (d.rendimiento_t_ha - loo_mean(d)).values)
            oof[b] += base(tr.iloc[b]) + m.predict(tr.iloc[b][cols])
        k[b] += 1
    p = oof / k
    return round(np.sqrt(mean_squared_error(y, p)), 3), round(mean_absolute_error(y, p), 3), round(r2_score(y, p), 3)

rk = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(tr))
gk = list(GroupKFold(n_splits=5).split(tr, groups=tr.pixel))
imp = lambda: SimpleImputer(strategy="median")
mods = {
    "Ridge (alpha alto)":      lambda: make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=np.logspace(1, 4, 20))),
    "SelectK10 + Ridge":       lambda: make_pipeline(imp(), SelectKBest(f_regression, k=10), StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 20))),
    "Random Forest":           lambda: make_pipeline(imp(), RandomForestRegressor(500, min_samples_leaf=5, max_features=0.33, random_state=SEED, n_jobs=-1)),
    "Extra Trees":             lambda: make_pipeline(imp(), ExtraTreesRegressor(500, min_samples_leaf=5, max_features=0.33, random_state=SEED, n_jobs=-1)),
    "Gradient Boosting":       lambda: make_pipeline(imp(), GradientBoostingRegressor(n_estimators=150, learning_rate=0.03, max_depth=2, subsample=0.7, random_state=SEED)),
}
sets = {"anomalias: satelital": sat_a, "anomalias: satelital+topo": sat_a + top_a,
        "anomalias: todo (sat+clima+topo)": anom}
filas = [("Linea base (media municipio)", "-", *cv(rk, None, []), *cv(gk, None, []))]
for sn, cols in sets.items():
    for mn, mk in mods.items():
        filas.append((mn, sn, *cv(rk, mk, cols), *cv(gk, mk, cols)))
out = pd.DataFrame(filas, columns=["modelo", "variables", "RMSE_aleat", "MAE_aleat", "R2_aleat", "RMSE_agrup", "MAE_agrup", "R2_agrup"])
print(out.sort_values("RMSE_agrup").to_string(index=False)); out.to_csv("resultados_modelos_ronda2.csv", index=False)
