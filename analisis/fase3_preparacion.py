"""
Reto AgroCebada 2026 - CRISP-DM Fase 3: Preparación de los datos
Genera una tabla con UNA fila por parcela (197) lista para modelar.

Uso:  python fase3_preparacion.py <carpeta DATASET_RETO_AGRO_2026 ya descomprimida> <salida.csv>
Requiere: pandas, numpy, geopandas, rasterio, pyproj
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer

BASE = sys.argv[1] if len(sys.argv) > 1 else "DATASET_RETO_AGRO_2026"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tabla_variables_agrocebada.csv"
ANIO = 2025                 # único ciclo con rendimiento (abril-octubre 2025)
NUB_MAX = 20                # % de nubosidad máximo aceptado
MESES = range(1, 11)        # enero-octubre
MESES_CLAVE = range(3, 8)   # marzo-julio (donde se observó señal)

# ---------------------------------------------------------------- 1. Parcelas
shp = f"{BASE}/Parcelas_Reto_AGC_CONJUNTO_70_30/Parcelas_Reto_AGC_CONJUNTO.shp"
g = gpd.read_file(shp).to_crs(4326)
cen = g.to_crs(32614).geometry.centroid.to_crs(4326)
meta = pd.DataFrame({
    "ID_POLIGONO": g["ID_POLIGON"], "Estado": g["Estado"], "Municipio": g["Municipio"],
    "lon": cen.x.values, "lat": cen.y.values,
})
y = pd.read_csv(f"{BASE}/ID_area_rendimiento_70_30_Reto_AgroCebada.csv").iloc[:, :4]
meta = meta.merge(y, on="ID_POLIGONO", how="left")
meta = meta.sort_values("ID_POLIGONO").reset_index(drop=True)
pts = list(zip(meta.lon, meta.lat))

# --------------------------------------------------------------- 2. Satelital
def mensual(df, cols, prefijo):
    """Media mensual por parcela de los índices (solo observaciones con nube < NUB_MAX)."""
    df = df.copy()
    df["f"] = pd.to_datetime(df.fecha_captura, format="%d/%m/%Y")
    df = df[(df.f.dt.year == ANIO) & (df.porcentaje_nubosidad < NUB_MAX) & df[cols[0]].notna()]
    df["mes"] = df.f.dt.month
    df = df[df.mes.isin(list(MESES))]   # solo ene-oct: noviembre y diciembre son posteriores a la cosecha
    t = df.groupby(["ID_POLIGONO", "mes"])[cols].mean().unstack("mes")
    t.columns = [f"{prefijo}_{c.replace('_promedio', '')}_m{m:02d}" for c, m in t.columns]
    return t

pro = pd.read_csv(f"{BASE}/Conjunto_datos_PRO_AgroCebada.csv")
bas = pd.read_csv(f"{BASE}/Conjunto_datos_BASICO_AgroCebada2026.csv")
bas = bas[bas.sensor == "Sentinel-2"]          # Landsat solo trae VI6T; se descarta

pro_cols = ["ndvi_promedio", "evi_promedio"]   # LAI y MSAVI son redundantes (r >= 0.96)
s2_cols = ["ndvi_promedio", "evi_promedio", "ndwi_promedio", "msi_promedio",
           "crc_promedio", "dswi2_promedio"]
T_pro = mensual(pro, pro_cols, "pl")
T_s2 = mensual(bas, s2_cols, "s2")
sat = T_pro.join(T_s2, how="outer").reindex(meta.ID_POLIGONO)

# Faltantes mensuales: interpolación lineal entre meses vecinos (máx. 1 mes seguido)
cols_sat = list(sat.columns)
n_faltantes_antes = int(sat[cols_sat].isna().sum().sum())
for pref in sorted({c.rsplit("_m", 1)[0] for c in cols_sat}):
    cs = [f"{pref}_m{m:02d}" for m in MESES if f"{pref}_m{m:02d}" in sat.columns]
    sat[cs] = sat[cs].interpolate(axis=1, limit=1, limit_area="inside")

# Variables derivadas de la curva (Planet NDVI y EVI, marzo-julio)
for idx in ("ndvi", "evi"):
    cs = [f"pl_{idx}_m{m:02d}" for m in MESES_CLAVE]
    v = sat[cs]
    sat[f"pl_{idx}_pico"] = v.max(axis=1)
    # mes del pico en abril-octubre (en marzo-julio siempre caía en julio y quedaba constante)
    vc = sat[[f"pl_{idx}_m{m:02d}" for m in range(4, 11)]]
    sat[f"pl_{idx}_mes_pico"] = vc.idxmax(axis=1).str[-2:].astype(float)
    sat[f"pl_{idx}_auc_mar_jul"] = v.sum(axis=1, min_count=3)
    sat[f"pl_{idx}_pend_abr_jun"] = sat[f"pl_{idx}_m06"] - sat[f"pl_{idx}_m04"]

# Heterogeneidad espacial dentro de la parcela (std de NDVI Planet, media mar-jul)
pro2 = pro.copy()
pro2["f"] = pd.to_datetime(pro2.fecha_captura, format="%d/%m/%Y")
pro2 = pro2[(pro2.f.dt.year == ANIO) & (pro2.f.dt.month.isin(MESES_CLAVE))
            & (pro2.porcentaje_nubosidad < NUB_MAX) & pro2.ndvi_std.notna()]
sat["pl_ndvi_std_mar_jul"] = pro2.groupby("ID_POLIGONO").ndvi_std.mean().reindex(meta.ID_POLIGONO)

# ------------------------------------------------------------------ 3. Clima
def muestrear(path, puntos):
    with rasterio.open(path) as r:
        v = np.array([x[0] for x in r.sample(puntos)], dtype=float)
        nd = r.nodata
    if nd is not None:
        v[v == nd] = np.nan
    v[v < -9000] = np.nan
    return v

CLIMA = {
    "prec": (f"{BASE}/Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/"
             "Reto_AgroCebada_CHIRPS_Precipitacion_2022_2025/Precipitacion_mensual/{a}/PREC_{a}_{m:02d}.tif"),
    "tmin": (f"{BASE}/Reto_AgroCebada_Temperatura_2022_2025_RECORTADA_CHIRPS/"
             "Reto_AgroCebada_Temperatura_2022_2025/Tmin/{a}/Tmin_{a}_{m:02d}.tif"),
    "tmax": (f"{BASE}/Reto_AgroCebada_Temperatura_2022_2025_RECORTADA_CHIRPS/"
             "Reto_AgroCebada_Temperatura_2022_2025/Tmax/{a}/Tmax_{a}_{m:02d}.tif"),
}
clima = pd.DataFrame(index=meta.index)
for k, plantilla in CLIMA.items():
    for m in MESES:
        clima[f"{k}_m{m:02d}"] = muestrear(plantilla.format(a=ANIO, m=m), pts)
for m in MESES:
    clima[f"tmed_m{m:02d}"] = (clima[f"tmin_m{m:02d}"] + clima[f"tmax_m{m:02d}"]) / 2
    clima[f"amp_m{m:02d}"] = clima[f"tmax_m{m:02d}"] - clima[f"tmin_m{m:02d}"]
ab_jun, jul_oct, ciclo = [4, 5, 6], [7, 8, 9, 10], list(range(4, 11))
clima["prec_abr_jun"] = clima[[f"prec_m{m:02d}" for m in ab_jun]].sum(axis=1)
clima["prec_jul_oct"] = clima[[f"prec_m{m:02d}" for m in jul_oct]].sum(axis=1)
clima["prec_ciclo"] = clima[[f"prec_m{m:02d}" for m in ciclo]].sum(axis=1)
for v in ("tmin", "tmax", "tmed", "amp"):
    clima[f"{v}_ciclo"] = clima[[f"{v}_m{m:02d}" for m in ciclo]].mean(axis=1)

# ------------------------------------------------------------- 4. Topografía
tb = f"{BASE}/Reto_AgroCebada_Topografia_INEGI_CEM4/" * 1
tdir = tb + "Reto_AgroCebada_Topografia_INEGI_CEM4/Reto_AgroCebada_Topografia_INEGI_CEM4/"
tr = Transformer.from_crs(4326, 6372, always_xy=True)   # el CEM viene en EPSG:6372
xy = [tr.transform(a, b) for a, b in pts]
topo = pd.DataFrame(index=meta.index)
topo["elevacion_m"] = muestrear(tdir + "Elevacion_INEGI_CEM4_120m.tif", xy)
topo["pendiente_grados"] = muestrear(tdir + "Pendiente_INEGI_CEM4_120m_grados.tif", xy)

# -------------------------------------------------------------- 5. Ensamblado
sat = sat.reset_index(drop=True)
tabla = pd.concat([meta, topo, clima, sat], axis=1)
tabla = tabla.rename(columns={"AREA_HA": "area_ha", "RENDIMIENTO_T_HA": "rendimiento_t_ha",
                              "CONJUNTO": "conjunto"})
tabla.to_csv(OUT, index=False, encoding="utf-8")

print(f"Tabla: {tabla.shape[0]} parcelas x {tabla.shape[1]} columnas -> {OUT}")
print(f"Faltantes satelitales mensuales antes de interpolar: {n_faltantes_antes}")
nan = tabla.isna().sum()
print("Columnas con faltantes tras limpieza:", nan[nan > 0].sort_values(ascending=False).head(15).to_dict())
