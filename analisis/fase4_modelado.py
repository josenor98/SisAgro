"""
Reto AgroCebada 2026 - CRISP-DM Fase 4/5: Modelado y evaluación (primera ronda)
Compara una línea base contra Ridge, Random Forest y Gradient Boosting con dos validaciones:
  - aleatoria (5 pliegues x 3 repeticiones)
  - agrupada por píxel climático (parcelas vecinas nunca quedan en entrenamiento y prueba a la vez)
Uso: python fase4_modelado.py tabla_variables_agrocebada.csv
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
SEED = 42
t = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "tabla_variables_agrocebada.csv")
tr = t[t.conjunto == "ENTRENAMIENTO"].reset_index(drop=True)
y = tr.rendimiento_t_ha.values

# píxel climático (rejilla CHIRPS 0.05°) -> grupo para la validación agrupada
tr["pixel"] = (np.floor((tr.lon + 99.94999880716205) / 0.05).astype(int).astype(str) + "_" +
               np.floor((21.39999957382679 - tr.lat) / 0.05).astype(int).astype(str))

num = tr.select_dtypes("number").columns.drop(["rendimiento_t_ha"])
grp = {
    "ubicacion": ["lat", "lon", "elevacion_m", "pendiente_grados", "area_ha"],
    "satelital": [c for c in num if c.startswith(("pl_", "s2_"))],
    "clima": [c for c in num if c.startswith(("prec", "tmin", "tmax", "tmed", "amp"))],
}
cat = ["Estado", "Municipio"]

def pipe(cols, modelo, usar_cat=True, escalar=False):
    n = Pipeline([("i", SimpleImputer(strategy="median"))] + ([("s", StandardScaler())] if escalar else []))
    trans = [("n", n, cols)] + ([("c", OneHotEncoder(handle_unknown="ignore"), cat)] if usar_cat else [])
    return Pipeline([("p", ColumnTransformer(trans)), ("m", modelo)])

def evaluar(make, X, splits):
    oof, n = np.zeros(len(y)), np.zeros(len(y))
    for a, b in splits:
        m = make(); m.fit(X.iloc[a], y[a]); oof[b] += m.predict(X.iloc[b]); n[b] += 1
    p = oof / n
    return np.sqrt(mean_squared_error(y, p)), mean_absolute_error(y, p), r2_score(y, p)

class MediaMunicipio:                      # línea base: media del municipio (o del estado/global si no hay)
    def fit(self, X, y):
        d = pd.DataFrame({"m": X.Municipio.values, "e": X.Estado.values, "y": y})
        self.m, self.e, self.g = d.groupby("m").y.mean(), d.groupby("e").y.mean(), y.mean(); return self
    def predict(self, X):
        return np.array([self.m.get(m, self.e.get(e, self.g)) for m, e in zip(X.Municipio, X.Estado)])

rk = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(tr))
gk = list(GroupKFold(n_splits=5).split(tr, groups=tr.pixel))
todas = sum(grp.values(), [])

modelos = {
    "Linea base (media municipio)": (lambda: MediaMunicipio(), cat),
    "Ridge": (lambda: pipe(todas, RidgeCV(alphas=np.logspace(-1, 3, 20)), escalar=True), todas + cat),
    "Random Forest": (lambda: pipe(todas, RandomForestRegressor(500, min_samples_leaf=2, max_features=0.33, random_state=SEED, n_jobs=-1)), todas + cat),
    "Gradient Boosting": (lambda: pipe(todas, GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=SEED)), todas + cat),
}
filas = []
for nom, (mk, cols) in modelos.items():
    for vn, sp in (("aleatoria", rk), ("agrupada por pixel", gk)):
        r = evaluar(mk, tr[cols], sp); filas.append((nom, vn, *r))
res = pd.DataFrame(filas, columns=["modelo", "validacion", "RMSE", "MAE", "R2"]).round(3)
print(res.to_string(index=False))
res.to_csv("resultados_modelos_ronda1.csv", index=False)

# Ablación con Random Forest (responde las preguntas guía): ¿qué grupo aporta más?
print("\nAblacion Random Forest (solo ese grupo + Estado/Municipio vs sin ellos)")
fila2 = []
combos = {"solo ubicacion": grp["ubicacion"], "solo satelital": grp["satelital"], "solo clima": grp["clima"],
          "satelital+clima": grp["satelital"] + grp["clima"], "todo": todas}
for nom, cols in combos.items():
    for usar_cat in (False, True):
        mk = lambda c=cols, u=usar_cat: pipe(c, RandomForestRegressor(500, min_samples_leaf=2, max_features=0.33, random_state=SEED, n_jobs=-1), usar_cat=u)
        X = tr[cols + (cat if usar_cat else [])]
        for vn, sp in (("aleatoria", rk), ("agrupada", gk)):
            r = evaluar(mk, X, sp); fila2.append((nom, "con Estado/Municipio" if usar_cat else "sin", vn, *r))
ab = pd.DataFrame(fila2, columns=["variables", "categoricas", "validacion", "RMSE", "MAE", "R2"]).round(3)
print(ab.to_string(index=False)); ab.to_csv("ablacion_rf_ronda1.csv", index=False)
