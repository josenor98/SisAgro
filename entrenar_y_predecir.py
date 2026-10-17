"""Entrena el modelo final, estima su error por validación cruzada y genera predicciones, pesos y combinaciones.
Salidas: predicciones_197.csv, predicciones_59_parcelas.csv, metricas.json, pesos_variables.csv, combinaciones.csv
Uso: python entrenar_y_predecir.py"""
import json
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from modelo import agregar_anomalias, ModeloAgroCebada, grupo_variable, SEED

t0 = pd.read_csv("tabla_variables_agrocebada.csv")
t, anom = agregar_anomalias(t0)
tr = t[t.conjunto == "ENTRENAMIENTO"].reset_index(drop=True)
y = tr.rendimiento_t_ha.values
tr["pixel"] = (np.floor((tr.lon + 99.94999880716205) / 0.05).astype(int).astype(str) + "_" +
               np.floor((21.39999957382679 - tr.lat) / 0.05).astype(int).astype(str))

def oof(splits, cols):
    o, b0, k = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for a, b in splits:
        m = ModeloAgroCebada(cols).fit(tr.iloc[a])
        b0[b] += m.base(tr.iloc[b]); o[b] += m.predict(tr.iloc[b]); k[b] += 1
    return o / k, b0 / k

def met(p):
    return {"RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 3),
            "MAE": round(float(mean_absolute_error(y, p)), 3), "R2": round(float(r2_score(y, p)), 3)}

rk = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(tr))
gk = list(GroupKFold(n_splits=5).split(tr, groups=tr.pixel))
p_rk, b_rk = oof(rk, anom); p_gk, b_gk = oof(gk, anom)
q10, q90 = np.quantile(y - p_rk, [0.10, 0.90])
final = ModeloAgroCebada(anom).fit(tr)
metricas = {"n_entrenamiento": int(len(tr)), "n_variables_modelo": len(anom), "alpha_ridge": round(final.alpha, 1),
            "modelo_final": {"aleatoria": met(p_rk), "agrupada_pixel": met(p_gk)},
            "linea_base_municipio": {"aleatoria": met(b_rk), "agrupada_pixel": met(b_gk)},
            "intervalo_80": {"q10": round(float(q10), 3), "q90": round(float(q90), 3)}}
json.dump(metricas, open("metricas.json", "w"), indent=2)
from modelo import media_loo
res_tr = (tr.rendimiento_t_ha - media_loo(tr)).values          # lo que la media del municipio no explica
pw = final.pesos()
pw["r_con_residuo"] = [np.corrcoef(tr[v + "_anom"].fillna(tr[v + "_anom"].median()), res_tr)[0, 1] for v in pw.variable]
pw.round(5).to_csv("pesos_variables.csv", index=False)
cor_all = final.correccion(t)
metricas["correccion_tipica_t_ha"] = {"desv_est": round(float(cor_all.std()), 3), "maxima_abs": round(float(np.abs(cor_all).max()), 3)}
json.dump(metricas, open("metricas.json", "w"), indent=2)

# combinaciones de grupos de variables (misma validación) -> pregunta guía 3
G = {g: [c for c in anom if grupo_variable(c) == g] for g in ["Satelital: Planet", "Satelital: Sentinel-2", "Clima", "Topografía y parcela"]}
sat, clm, top = G["Satelital: Planet"] + G["Satelital: Sentinel-2"], G["Clima"], G["Topografía y parcela"]
combos = {"Solo municipio (línea base)": [], "Municipio + satelital": sat, "Municipio + clima": clm,
          "Municipio + topografía y parcela": top, "Municipio + satelital + topografía": sat + top,
          "Municipio + satelital + clima": sat + clm, "Municipio + todas las variables": sat + clm + top}
filas = []
for nom, cols in combos.items():
    r1, r2 = met(oof(rk, cols)[0]), met(oof(gk, cols)[0])
    filas.append((nom, len(cols), r1["RMSE"], r1["MAE"], r1["R2"], r2["RMSE"], r2["MAE"], r2["R2"]))
pd.DataFrame(filas, columns=["combinacion", "n_variables", "RMSE_aleat", "MAE_aleat", "R2_aleat",
                             "RMSE_agrup", "MAE_agrup", "R2_agrup"]).to_csv("combinaciones.csv", index=False)

pred = final.predict(t)
out = t[["ID_POLIGONO", "Estado", "Municipio", "conjunto", "area_ha", "lat", "lon", "rendimiento_t_ha"]].copy()
out["rendimiento_predicho"] = pred
idx = out.index[out.conjunto == "ENTRENAMIENTO"]
out.loc[idx, "rendimiento_predicho"] = pd.Series(p_rk, index=tr.ID_POLIGONO).reindex(out.loc[idx, "ID_POLIGONO"]).values
out["p10"] = out.rendimiento_predicho + q10
out["p90"] = out.rendimiento_predicho + q90
out["produccion_esperada_t"] = out.rendimiento_predicho * out.area_ha
out.round(3).to_csv("predicciones_197.csv", index=False)
pr = out[out.conjunto == "PREDICCION"][["ID_POLIGONO", "rendimiento_predicho"]].round(3)
pr.rename(columns={"rendimiento_predicho": "RENDIMIENTO_T_HA_PREDICHO"}).to_csv("predicciones_59_parcelas.csv", index=False)
print(json.dumps(metricas, indent=1)); print(pd.read_csv("combinaciones.csv").to_string(index=False))
print(final.pesos().head(8)[["etiqueta", "coeficiente", "peso_pct"]].round(3).to_string())
print("59 predicciones:", pr.rendimiento_predicho.describe().round(2).to_dict())
