"""SisAgro - Sistema de análisis y predicción de rendimiento de cebada (Reto AgroCebada FIRA 2026)."""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import asesor as ase
from estilo import CSS, ESCALA, P, bloque_asesor, bloque_ia_html, como_leer, encabezado, marca_lateral, registrar_tema
from modelo import agregar_anomalias, ModeloAgroCebada, Zonas, predecir_nuevas, etiqueta

st.set_page_config(page_title="SisAgro | Predicción de rendimiento de cebada", page_icon=Image.open("assets/favicon.png"), layout="wide")
registrar_tema()
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
LOGO = open("assets/logo.png", "rb").read()
MESES_TXT = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct"]


# ------------------------------------------------------------------ carga de datos y modelo
@st.cache_data
def cargar():
    t, anom = agregar_anomalias(pd.read_csv("tabla_variables_agrocebada.csv"))
    return (t, anom, pd.read_csv("predicciones_197.csv"), json.load(open("metricas.json")),
            pd.read_csv("pesos_variables.csv"), pd.read_csv("combinaciones.csv"))


@st.cache_resource
def modelo_final(_t, anom):
    return ModeloAgroCebada(anom).fit(_t[_t.conjunto == "ENTRENAMIENTO"]), Zonas(_t)


t, anom, pred, met, pesos, combos = cargar()
modelo, zonas = modelo_final(t, anom)
Q10, Q90 = met["intervalo_80"]["q10"], met["intervalo_80"]["q90"]
REF = ase.referencia(t)
ESCALA_ANOM = {c: float(t[c].std()) for c in anom}

# IA generativa opcional: solo se activa si hay una clave en los secretos de Streamlit
try:
    API = st.secrets["ANTHROPIC_API_KEY"]
    MODELO_IA = st.secrets.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
except Exception:
    API, MODELO_IA = None, None


def bloque_ia(lect, clave):
    """Botón que pide a un modelo de lenguaje una lectura ampliada basada en los hechos de la lectura del asesor."""
    if not (API and st.session_state.get("usar_ia")):
        return
    cache = st.session_state.setdefault("ia_cache", {})
    ctx = ase.a_texto(lect); k = ase.clave_contexto(ctx)
    if st.button("Generar lectura ampliada con IA", key=f"ia_{clave}"):
        if k not in cache:
            if st.session_state.get("n_ia", 0) >= 12:
                st.warning("Se alcanzó el límite de lecturas con IA de esta sesión.")
                return
            with st.spinner("Redactando la lectura..."):
                txt, err = ase.redactar_con_ia(ctx, API, MODELO_IA)
            st.session_state["n_ia"] = st.session_state.get("n_ia", 0) + 1
            cache[k] = txt if txt else f"No fue posible generar la lectura ampliada. {err}"
    if k in cache:
        st.markdown(bloque_ia_html(cache[k]), unsafe_allow_html=True)


def mostrar_asesor(lect, clave):
    if lect:
        st.markdown(bloque_asesor(lect), unsafe_allow_html=True)
        bloque_ia(lect, clave)


def grafica_contribuciones(contrib, n=8):
    c = contrib.head(n).iloc[::-1]
    f = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=[P["cebada"] if v >= 0 else P["arcilla"] for v in c.values]))
    f.update_layout(height=330, margin=dict(l=0, r=10, t=40, b=0), title="Aporte de cada variable al rendimiento esperado (t/ha)")
    return f


# ------------------------------------------------------------------ encabezado y filtros
st.markdown(encabezado(LOGO, len(t), t.Estado.nunique()), unsafe_allow_html=True)

with st.sidebar:
    st.markdown(marca_lateral(LOGO), unsafe_allow_html=True)
    st.header("Filtros")
    estados = st.multiselect("Estado", sorted(pred.Estado.unique()), default=sorted(pred.Estado.unique()))
    grupo = st.radio("Parcelas", ["A predecir (reto)", "Entrenamiento (con rendimiento real)", "Todas"], index=0)
    st.caption("En las parcelas de entrenamiento se muestra la predicción fuera de muestra (validación cruzada), "
               "no la que el modelo ajusta sobre sí mismo.")
    if API:
        st.markdown("---")
        st.checkbox("Ampliar lecturas con IA generativa", key="usar_ia", help="Un modelo de lenguaje redacta una lectura más completa con los resultados del sistema.")

vista = pred[pred.Estado.isin(estados)].copy()
if grupo.startswith("A predecir"):
    vista = vista[vista.conjunto == "PREDICCION"]
elif grupo.startswith("Entrenamiento"):
    vista = vista[vista.conjunto == "ENTRENAMIENTO"]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Panorama", "Consulta por parcela", "Predecir parcela nueva",
                                              "Simulador", "Modelo y pesos", "Uso financiero"])

# ------------------------------------------------------------------ TAB 1: panorama
with tab1:
    if vista.empty:
        st.warning("No hay parcelas con los filtros seleccionados.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parcelas", f"{len(vista)}")
        c2.metric("Rendimiento esperado medio", f"{vista.rendimiento_predicho.mean():.2f} t/ha")
        c3.metric("Producción esperada", f"{vista.produccion_esperada_t.sum():,.0f} t")
        c4.metric("Superficie", f"{vista.area_ha.sum():,.0f} ha")

        st.subheader("Mapa de rendimiento esperado")
        vista["real"] = vista.rendimiento_t_ha.round(2)
        fig = px.scatter_map(
            vista, lat="lat", lon="lon", color="rendimiento_predicho", size="area_ha", size_max=18,
            color_continuous_scale=ESCALA, range_color=(2.0, 5.0), zoom=9, height=520, hover_name="ID_POLIGONO",
            hover_data={"Estado": True, "Municipio": True, "rendimiento_predicho": ":.2f", "area_ha": ":.1f",
                        "real": True, "lat": False, "lon": False},
            labels={"rendimiento_predicho": "Rendimiento esperado (t/ha)", "area_ha": "Área (ha)", "real": "Real (t/ha)"})
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), map_style="carto-positron")
        st.plotly_chart(fig, width="stretch")
        st.markdown(como_leer(ase.EXPLICA["mapa"]), unsafe_allow_html=True)

        colA, colB = st.columns([3, 2])
        with colA:
            st.subheader("Predicciones por parcela")
            tabla = vista[["ID_POLIGONO", "Estado", "Municipio", "area_ha", "rendimiento_predicho", "p10", "p90",
                           "produccion_esperada_t", "rendimiento_t_ha"]].rename(columns={
                "ID_POLIGONO": "Parcela", "area_ha": "Área (ha)", "rendimiento_predicho": "Esperado (t/ha)",
                "p10": "Mínimo 80 %", "p90": "Máximo 80 %", "produccion_esperada_t": "Producción (t)",
                "rendimiento_t_ha": "Real (t/ha)"}).round(2)
            st.dataframe(tabla, hide_index=True, width="stretch", height=340)
            st.download_button("Descargar CSV", tabla.to_csv(index=False).encode("utf-8"), "predicciones_sisagro.csv", "text/csv")
            st.caption(f"El rango de 80 % se obtiene de los errores de validación cruzada ({Q10:+.2f} a {Q90:+.2f} t/ha alrededor de la estimación).")
        with colB:
            st.subheader("Rendimiento por estado")
            fb = px.box(vista, x="Estado", y="rendimiento_predicho", color="Estado", points="all",
                        color_discrete_sequence=[P["bosque"], P["trigo"], P["tierra"]],
                        labels={"rendimiento_predicho": "Rendimiento esperado (t/ha)"})
            fb.update_layout(showlegend=False, height=340, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fb, width="stretch")
            st.markdown(como_leer(ase.EXPLICA["estados"]), unsafe_allow_html=True)
        mostrar_asesor(ase.lectura_mapa(vista, REF), "mapa")

# ------------------------------------------------------------------ TAB 2: consulta por parcela
with tab2:
    ids = sorted(vista.ID_POLIGONO) if not vista.empty else sorted(pred.ID_POLIGONO)
    pid = st.selectbox("Parcela", ids, key="parcela_consulta")
    d1 = t[t.ID_POLIGONO == pid]
    fila = d1.iloc[0]
    pr = pred[pred.ID_POLIGONO == pid].iloc[0]
    refmask = t._ref == fila._ref
    base_p, corr_p = float(modelo.base(d1)[0]), float(modelo.correccion(d1)[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ubicación", f"{fila.Municipio}", f"{fila.Estado}", delta_color="off")
    c2.metric("Rendimiento esperado", f"{pr.rendimiento_predicho:.2f} t/ha", f"rango 80 %: {pr.p10:.2f} a {pr.p90:.2f}", delta_color="off")
    c3.metric("Producción esperada", f"{pr.produccion_esperada_t:.1f} t", f"{fila.area_ha:.1f} ha", delta_color="off")
    if pd.notna(fila.rendimiento_t_ha):
        c4.metric("Rendimiento real", f"{fila.rendimiento_t_ha:.2f} t/ha", f"diferencia {pr.rendimiento_predicho - fila.rendimiento_t_ha:+.2f}", delta_color="off")
    else:
        c4.metric("Rendimiento real", "oculto por el reto")
    st.caption(f"Cómo se arma la estimación: promedio de {fila.Municipio} ({base_p:.2f} t/ha) más el ajuste por las variables de la parcela ({corr_p:+.2f} t/ha). "
               "El modelo se ejecuta en vivo al elegir la parcela.")

    def serie(prefijo, titulo):
        cols = [f"{prefijo}_m{m:02d}" for m in range(1, 11)]
        f = go.Figure()
        f.add_trace(go.Scatter(x=MESES_TXT, y=fila[cols].astype(float), name=f"Parcela {pid}", line=dict(color=P["bosque"], width=3), mode="lines+markers"))
        f.add_trace(go.Scatter(x=MESES_TXT, y=t.loc[refmask, cols].mean(), name="Promedio de su zona", line=dict(color=P["trigo"], width=2.5, dash="dash")))
        f.add_vrect(x0="Abr", x1="Oct", fillcolor=P["crema2"], opacity=0.55, line_width=0, layer="below")
        f.update_layout(title=titulo, height=310, margin=dict(l=0, r=0, t=44, b=0), legend=dict(orientation="h", y=-0.2))
        return f

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(serie("pl_evi", "EVI mensual (Planet, 2025)"), width="stretch")
        st.markdown(como_leer(ase.EXPLICA["evi"]), unsafe_allow_html=True)
    with g2:
        st.plotly_chart(serie("s2_ndvi", "NDVI mensual (Sentinel-2, 2025)"), width="stretch")
        st.markdown(como_leer(ase.EXPLICA["ndvi"]), unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        fp = go.Figure(go.Bar(x=MESES_TXT, y=fila[[f"prec_m{m:02d}" for m in range(1, 11)]].astype(float), marker_color=P["agua"]))
        fp.update_layout(title="Lluvia mensual (CHIRPS, mm)", height=310, margin=dict(l=0, r=0, t=44, b=0))
        st.plotly_chart(fp, width="stretch")
        st.markdown(como_leer(ase.EXPLICA["lluvia"]), unsafe_allow_html=True)
    with h2:
        ft = go.Figure()
        ft.add_trace(go.Scatter(x=MESES_TXT, y=fila[[f"tmax_m{m:02d}" for m in range(1, 11)]].astype(float), name="Tmax", line=dict(color=P["arcilla"], width=3)))
        ft.add_trace(go.Scatter(x=MESES_TXT, y=fila[[f"tmin_m{m:02d}" for m in range(1, 11)]].astype(float), name="Tmin", line=dict(color=P["agua"], width=3)))
        ft.update_layout(title="Temperatura mensual (CHIRTS-ERA5, °C)", height=310, margin=dict(l=0, r=0, t=44, b=0), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(ft, width="stretch")
        st.markdown(como_leer(ase.EXPLICA["temp"]), unsafe_allow_html=True)

    contrib = modelo.contribuciones(d1)
    st.plotly_chart(grafica_contribuciones(contrib), width="stretch")
    st.markdown(como_leer(ase.EXPLICA["contrib"]), unsafe_allow_html=True)
    real = float(fila.rendimiento_t_ha) if pd.notna(fila.rendimiento_t_ha) else None
    mostrar_asesor(ase.lectura_parcela(pid, float(pr.rendimiento_predicho), float(pr.p10), float(pr.p90), base_p, corr_p, float(fila.area_ha),
                                       fila.Municipio, fila.Estado, contrib, fila, fila, ESCALA_ANOM, REF, real=real), f"parcela_{pid}")

# ------------------------------------------------------------------ TAB 3: predecir parcela nueva
with tab3:
    st.subheader("Predecir el rendimiento de una parcela nueva")
    st.write("Captura los datos de una parcela, o sube un archivo con varias, y el modelo estima su rendimiento. "
             "Lo único obligatorio es el municipio; cada variable que dejes en el promedio de su zona no aporta ajuste.")
    modo = st.radio("Forma de ingresar los datos", ["Captura manual", "Subir archivo CSV"], horizontal=True)
    top_vars = [v for v in pesos.variable if v != "area_ha"][:10]

    if modo == "Captura manual":
        municipios = sorted(zonas.mun_estado)
        mun_sel = st.selectbox("Municipio", municipios + ["Otro municipio"], key="nuevo_mun")
        if mun_sel == "Otro municipio":
            mun = st.text_input("Nombre del municipio", key="nuevo_mun_txt")
            est = st.selectbox("Estado", ["Puebla", "Hidalgo", "Tlaxcala"], key="nuevo_est")
        else:
            mun, est = mun_sel, zonas.mun_estado[mun_sel]
            st.caption(f"Estado: {est}. Promedio histórico del municipio: {float(modelo.base(pd.DataFrame({'Municipio': [mun], 'Estado': [est]}))[0]):.2f} t/ha.")
        ref = zonas.referencia(mun, est)
        area = st.number_input("Superficie de la parcela (ha)", 0.1, 500.0, 5.0, 0.5, key="nuevo_area")
        st.markdown("**Variables con más peso en el modelo.** Escribe el valor real; por defecto aparece el promedio de la zona.")
        cols = st.columns(2)
        vals = {}
        for i, v in enumerate(top_vars):
            dflt = float(ref[v]) if v in ref.index else 0.0
            vals[v] = cols[i % 2].number_input(etiqueta(v), value=round(dflt, 4), format="%.4f", key=f"man_{mun}_{v}", help=f"Promedio de la zona: {dflt:.4f}")
        if st.button("Predecir rendimiento", type="primary", key="btn_manual"):
            if not mun:
                st.warning("Escribe el nombre del municipio.")
            else:
                nueva = pd.DataFrame([{**vals, "Municipio": mun, "Estado": est, "area_ha": area}])
                res, X = predecir_nuevas(modelo, zonas, nueva, Q10, Q90)
                r = res.iloc[0]
                a, b, c, d = st.columns(4)
                a.metric("Rendimiento esperado", f"{r.rendimiento_predicho:.2f} t/ha")
                b.metric("Rango 80 %", f"{r.p10:.2f} a {r.p90:.2f}")
                c.metric("Producción esperada", f"{r.produccion_esperada_t:.1f} t")
                d.metric("Promedio del municipio", f"{r.media_municipio:.2f} t/ha", f"ajuste {r.correccion:+.2f}", delta_color="off")
                contrib = modelo.contribuciones(X)
                st.plotly_chart(grafica_contribuciones(contrib), width="stretch")
                st.markdown(como_leer(ase.EXPLICA["contrib"]), unsafe_allow_html=True)
                lect = ase.lectura_parcela(None, float(r.rendimiento_predicho), float(r.p10), float(r.p90), float(r.media_municipio), float(r.correccion),
                                           float(area), mun, est, contrib, X.iloc[0], nueva.iloc[0], ESCALA_ANOM, REF, mun_conocido=(mun_sel != "Otro municipio"))
                lect["avisos"].append("Las variables que no capturaste se tomaron como el promedio de su zona, por lo que sus señales no se comparan.")
                mostrar_asesor(lect, "manual")
    else:
        plantilla = t[t.conjunto == "PREDICCION"].groupby("Estado").head(1)[["ID_POLIGONO", "Estado", "Municipio", "area_ha"] + top_vars]
        st.download_button("Descargar plantilla de ejemplo", plantilla.round(4).to_csv(index=False).encode("utf-8"), "plantilla_parcelas.csv", "text/csv")
        st.caption("Columnas: Municipio (obligatoria), Estado, ID_POLIGONO, area_ha y cualquier variable de la tabla con los mismos nombres del diccionario de datos. "
                   "Las que falten se toman como el promedio de la zona.")
        arch = st.file_uploader("Archivo CSV con las parcelas", type=["csv"])
        if arch is not None:
            nuevo = pd.read_csv(arch)
            if "Municipio" not in nuevo.columns:
                st.error("El archivo debe tener la columna Municipio.")
            else:
                res, _ = predecir_nuevas(modelo, zonas, nuevo, Q10, Q90)
                ident = nuevo[["ID_POLIGONO"]].reset_index(drop=True) if "ID_POLIGONO" in nuevo.columns else pd.DataFrame()
                sal = pd.concat([ident, res], axis=1).round(2)
                st.dataframe(sal, hide_index=True, width="stretch")
                st.download_button("Descargar predicciones", sal.to_csv(index=False).encode("utf-8"), "predicciones_nuevas.csv", "text/csv")
                vv = res.copy()
                vv["ID_POLIGONO"] = range(len(vv))
                if "produccion_esperada_t" not in vv: vv["produccion_esperada_t"] = 0.0
                mostrar_asesor(ase.lectura_mapa(vv, REF), "csv")

# ------------------------------------------------------------------ TAB 4: simulador
with tab4:
    st.subheader("Simulador de escenarios")
    st.write("Parte de una parcela real, cambia su municipio o sus variables y vuelve a ejecutar el modelo. "
             "Los deslizadores son diferencias respecto al promedio de su zona.")
    sp = st.selectbox("Parcela base", sorted(t.ID_POLIGONO), key="parcela_sim")
    f0 = t[t.ID_POLIGONO == sp].iloc[[0]].copy()
    municipios = sorted(zonas.mun_estado)
    mun_nuevo = st.selectbox("Municipio", municipios, index=municipios.index(f0.Municipio.iloc[0]))
    top = [v + "_anom" for v in pesos.variable.head(6)]
    cols = st.columns(2)
    f1 = f0.copy()
    f1["Municipio"], f1["Estado"] = mun_nuevo, zonas.mun_estado[mun_nuevo]
    for i, v in enumerate(top):
        sd = float(t[v].std()); v0 = float(f0[v].iloc[0]); lim = round(3 * sd, 3)
        f1[v] = cols[i % 2].slider(f"{etiqueta(v)} (frente a su zona)", -lim, lim, float(np.clip(v0, -lim, lim)), step=round(sd / 20, 4), key=f"sl_{sp}_{v}")
    p0, p1 = float(modelo.predict(f0)[0]), float(modelo.predict(f1)[0])
    a, b, c = st.columns(3)
    a.metric("Estimación original", f"{p0:.2f} t/ha")
    b.metric("Estimación del escenario", f"{p1:.2f} t/ha", f"{p1 - p0:+.2f} t/ha")
    c.metric("Promedio del municipio elegido", f"{float(modelo.base(f1)[0]):.2f} t/ha")
    mostrar_asesor(ase.lectura_simulador(p0, p1, f0.Municipio.iloc[0], mun_nuevo, float(modelo.base(f0)[0]), float(modelo.base(f1)[0]), REF), "sim")

# ------------------------------------------------------------------ TAB 5: modelo y pesos
with tab5:
    st.subheader("Cómo funciona el modelo")
    ct = met["correccion_tipica_t_ha"]
    st.markdown(
        "**1. Homogeneización.** Cada variable se expresa como su diferencia respecto al promedio de su zona (anomalía), "
        "se imputan faltantes con la mediana y se estandariza (z-score), de modo que todas quedan en la misma escala.  \n"
        f"**2. Pesos.** Una regresión Ridge asigna un peso a cada una de las {met['n_variables_modelo']} variables.  \n"
        "**3. Estimación = promedio del municipio + ajuste ponderado.** "
        f"El ajuste típico es de ±{ct['desv_est']:.2f} t/ha (máximo {ct['maxima_abs']:.2f}).")
    m = met
    tabm = pd.DataFrame({
        "Modelo": ["Línea base (promedio del municipio)", "Modelo final (Ridge ponderado)"],
        "RMSE aleatoria": [m["linea_base_municipio"]["aleatoria"]["RMSE"], m["modelo_final"]["aleatoria"]["RMSE"]],
        "MAE aleatoria": [m["linea_base_municipio"]["aleatoria"]["MAE"], m["modelo_final"]["aleatoria"]["MAE"]],
        "R² aleatoria": [m["linea_base_municipio"]["aleatoria"]["R2"], m["modelo_final"]["aleatoria"]["R2"]],
        "RMSE agrupada": [m["linea_base_municipio"]["agrupada_pixel"]["RMSE"], m["modelo_final"]["agrupada_pixel"]["RMSE"]],
        "R² agrupada": [m["linea_base_municipio"]["agrupada_pixel"]["R2"], m["modelo_final"]["agrupada_pixel"]["R2"]]})
    st.dataframe(tabm, hide_index=True, width="stretch")
    st.caption("Validación aleatoria: 5 pliegues por 3 repeticiones. Agrupada: parcelas de un mismo píxel climático nunca quedan en entrenamiento y prueba a la vez. Errores en t/ha, con 138 parcelas de rendimiento conocido.")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Peso por tipo de variable (% del total)**")
        pg = pesos.groupby("grupo").peso_pct.sum().sort_values().reset_index()
        fg = px.bar(pg, x="peso_pct", y="grupo", orientation="h", color_discrete_sequence=[P["bosque"]], labels={"peso_pct": "% del peso", "grupo": ""})
        fg.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0)); st.plotly_chart(fg, width="stretch")
        st.markdown(como_leer(ase.EXPLICA["pesos_grupo"]), unsafe_allow_html=True)
        st.markdown("**Peso por índice (suma de sus meses)**")
        pi = pesos.dropna(subset=["familia_indice"]).groupby("familia_indice").peso_pct.sum().sort_values().reset_index()
        fi = px.bar(pi, x="peso_pct", y="familia_indice", orientation="h", color_discrete_sequence=[P["trigo"]], labels={"peso_pct": "% del peso", "familia_indice": ""})
        fi.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0)); st.plotly_chart(fi, width="stretch")
        st.markdown(como_leer(ase.EXPLICA["pesos_indice"]), unsafe_allow_html=True)
    with cB:
        st.markdown("**Las 15 variables con más peso**")
        p15 = pesos.head(15).iloc[::-1]
        fv = px.bar(p15, x="coeficiente", y="etiqueta", orientation="h", color=p15.coeficiente > 0,
                    color_discrete_map={True: P["cebada"], False: P["arcilla"]}, labels={"coeficiente": "Coeficiente", "etiqueta": ""})
        fv.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fv, width="stretch")
        st.markdown(como_leer(ase.EXPLICA["pesos_top"]), unsafe_allow_html=True)
    with st.expander("Ver el peso de las 74 variables"):
        tp = pesos[["etiqueta", "grupo", "coeficiente", "peso_pct", "r_con_residuo"]].rename(columns={
            "etiqueta": "Variable", "grupo": "Grupo", "coeficiente": "Coeficiente", "peso_pct": "Peso (%)",
            "r_con_residuo": "Correlación con lo que no explica el municipio"}).round(3)
        st.dataframe(tp, hide_index=True, width="stretch")

    st.subheader("Qué combinación de variables funciona mejor")
    cb = combos[["combinacion", "n_variables", "RMSE_aleat", "MAE_aleat", "RMSE_agrup", "R2_agrup"]].rename(columns={
        "combinacion": "Combinación", "n_variables": "Variables", "RMSE_aleat": "RMSE aleatoria", "MAE_aleat": "MAE aleatoria",
        "RMSE_agrup": "RMSE agrupada", "R2_agrup": "R² agrupada"})
    st.dataframe(cb, hide_index=True, width="stretch")
    st.markdown(como_leer(ase.EXPLICA["combos"]), unsafe_allow_html=True)

    st.subheader("Preguntas guía del reto")
    top_ind = pesos.dropna(subset=["familia_indice"]).groupby("familia_indice").peso_pct.sum().sort_values(ascending=False)
    clima = pesos[pesos.grupo == "Clima"].head(3).etiqueta.tolist()
    mejor = combos.sort_values("RMSE_agrup").iloc[0]
    st.markdown(f"""
**1. Qué índice es más importante.** Por peso en el modelo, **{top_ind.index[0]}** ({top_ind.iloc[0]:.0f} % del peso total), seguido de {top_ind.index[1]} ({top_ind.iloc[1]:.0f} %) y {top_ind.index[2]} ({top_ind.iloc[2]:.0f} %). Sin controlar la ubicación, el EVI de Planet de mayo es el más correlacionado con el rendimiento (r cercana a 0.60); dentro de cada municipio la relación baja a |r| de 0.26 como máximo.

**2. Qué variable climática tiene mayor peso.** Dentro del modelo, **{clima[0]}**, seguida de {clima[1]} y {clima[2]}. Hay que leerlo con cuidado: el clima tiene resolución de unos 5 km (25 píxeles para 197 parcelas), así que gran parte de su señal es ubicación.

**3. Qué combinación funciona mejor.** La menor RMSE en validación agrupada la obtiene «{mejor.combinacion}» ({mejor.RMSE_agrup:.3f} t/ha). Las diferencias entre combinaciones son de unos 0.01 t/ha, dentro del ruido: con un solo ciclo y 138 parcelas, la ubicación explica casi todo lo predecible.
""")
    mostrar_asesor(ase.lectura_modelo(met, pred, pesos, combos, REF), "modelo")

# ------------------------------------------------------------------ TAB 6: uso financiero
with tab6:
    st.subheader("Uso de las estimaciones en inclusión financiera")
    st.write("Una estimación de rendimiento antes de cosecha ayuda a evaluar la capacidad productiva de una parcela. "
             "Define un rendimiento mínimo de referencia, por ejemplo el necesario para cubrir costos, y revisa cuántas parcelas "
             "lo superan con holgura, cuáles son inciertas y cuáles están en riesgo.")
    umbral = st.slider("Rendimiento mínimo de referencia (t/ha)", 2.0, 5.0, 3.5, 0.1)
    base = pred[pred.Estado.isin(estados)]
    base = base[base.conjunto == "PREDICCION"] if grupo.startswith("A predecir") else vista
    if base.empty:
        st.warning("No hay parcelas con los filtros seleccionados.")
    else:
        base = base.copy()
        orden = ["Holgada (rango 80 % sobre la referencia)", "Incierta (el rango cruza la referencia)", "En riesgo (rango 80 % bajo la referencia)"]
        base["Clasificación"] = np.select([base.p10 >= umbral, base.p90 < umbral], [orden[0], orden[2]], default=orden[1])
        cnt = base.groupby(["Estado", "Clasificación"]).size().reset_index(name="Parcelas")
        fc = px.bar(cnt, x="Estado", y="Parcelas", color="Clasificación", category_orders={"Clasificación": orden},
                    color_discrete_map={orden[0]: "#3F7D2E", orden[1]: P["trigo"], orden[2]: P["arcilla"]})
        fc.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fc, width="stretch")
        st.markdown(como_leer(ase.EXPLICA["finanzas"]), unsafe_allow_html=True)
        pct = base.Clasificación.value_counts(normalize=True).reindex(orden).fillna(0) * 100
        k1, k2, k3 = st.columns(3)
        k1.metric("Holgada", f"{pct[orden[0]]:.0f} %"); k2.metric("Incierta", f"{pct[orden[1]]:.0f} %"); k3.metric("En riesgo", f"{pct[orden[2]]:.0f} %")
        mostrar_asesor(ase.lectura_financiera(base, umbral, orden), "fin")
