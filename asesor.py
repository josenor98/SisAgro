"""Asesor SisAgro: convierte los resultados del modelo y de las gráficas en lecturas en lenguaje claro
(qué significan, si el rendimiento es bueno o bajo y qué hacer para mejorar).

Funciona en dos niveles:
  1) Motor de interpretación (siempre activo): reglas explícitas sobre las salidas del modelo. No inventa cifras:
     todo número que muestra sale de los datos y del modelo.
  2) Redacción con IA generativa (opcional): si se configura una clave de API, un modelo de lenguaje redacta una lectura
     ampliada usando únicamente los hechos que calcula el motor.
"""
import hashlib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ referencias y clasificación
def referencia(t):
    y = t.loc[t.conjunto == "ENTRENAMIENTO", "rendimiento_t_ha"]
    return {"media": float(y.mean()), "sd": float(y.std()), "min": float(y.min()), "max": float(y.max())}


def nivel(rend, ref):
    if rend >= ref["media"] + 0.5 * ref["sd"]: return "Alto"
    if rend <= ref["media"] - 0.5 * ref["sd"]: return "Bajo"
    return "Medio"


VEREDICTO = {"Alto": "Bueno", "Medio": "Regular", "Bajo": "Bajo"}


def umbrales(ref):
    return ref["media"] - 0.5 * ref["sd"], ref["media"] + 0.5 * ref["sd"]


# ------------------------------------------------------------------ señales de una parcela frente a su zona
VIGOR_COLS = ["s2_ndvi_m03_anom", "s2_ndvi_m04_anom", "s2_ndvi_m05_anom", "s2_evi_m03_anom", "s2_evi_m04_anom",
              "s2_evi_m05_anom", "pl_ndvi_m04_anom", "pl_ndvi_m05_anom", "pl_evi_m04_anom", "pl_evi_m05_anom"]


def _z(anom_row, escala, c):
    if c in anom_row.index and escala.get(c): return float(anom_row[c]) / escala[c]
    return np.nan


def _prom(vals):
    v = [x for x in vals if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


def _cat(z, mas="por encima de", menos="por debajo de", igual="similar a"):
    if np.isnan(z): return None
    return mas if z >= 0.5 else menos if z <= -0.5 else igual


def senales(anom_row, raw_row, escala):
    z = lambda c: _z(anom_row, escala, c)
    s = {"vigor": _prom([z(c) for c in VIGOR_COLS]),
         "humedad": _prom([-z("s2_msi_m04_anom"), -z("s2_msi_m05_anom"), z("s2_ndwi_m04_anom"), z("s2_ndwi_m05_anom")]),
         "hetero": z("pl_ndvi_std_mar_jul_anom"), "pend_z": z("pendiente_grados_anom"),
         "pendiente": float(raw_row["pendiente_grados"]) if "pendiente_grados" in raw_row.index and pd.notna(raw_row["pendiente_grados"]) else np.nan,
         "lluvia_pct": np.nan}
    if "prec_abr_jun_anom" in anom_row.index and "prec_abr_jun" in raw_row.index:
        zona = float(raw_row["prec_abr_jun"]) - float(anom_row["prec_abr_jun_anom"])
        if zona > 0: s["lluvia_pct"] = float(anom_row["prec_abr_jun_anom"]) / zona * 100
    return s


def _items_senales(s):
    it = []
    c = _cat(s["vigor"])
    if c: it.append(f"<b>Vigor del cultivo</b> (índices de vegetación de febrero a mayo): {c} su zona. "
                    + ("Su cultivo arrancó con menos desarrollo que el de sus vecinas." if c.startswith("por debajo") else
                       "Su cultivo arrancó con más desarrollo que el de sus vecinas." if c.startswith("por encima") else
                       "Su desarrollo inicial es parecido al de sus vecinas."))
    c = _cat(s["humedad"], "mayor", "menor", "similar")
    if c: it.append(f"<b>Humedad en la vegetación</b> (índices MSI y NDWI de abril y mayo): {c} disponibilidad de agua que su zona."
                    if c != "similar" else "<b>Humedad en la vegetación</b> (índices MSI y NDWI de abril y mayo): similar a la de su zona.")
    if not np.isnan(s["lluvia_pct"]): it.append(f"<b>Lluvia de abril a junio</b>: {s['lluvia_pct']:+.0f} % frente al promedio de su zona (dato de 5 km, compartido con parcelas vecinas).")
    c = _cat(s["hetero"], "más desigual", "más uniforme", "similar")
    if c: it.append("<b>Uniformidad dentro de la parcela</b> (variación del NDVI entre sus píxeles): "
                    + ("más desigual que la de su zona." if c == "más desigual" else "más uniforme que la de su zona." if c == "más uniforme" else "similar a la de su zona."))
    if not np.isnan(s["pendiente"]): it.append(f"<b>Pendiente del terreno</b>: {s['pendiente']:.1f}°"
                                              + (" (más inclinada que la mayoría de su zona)." if (s["pend_z"] >= 1 if not np.isnan(s["pend_z"]) else False) else "."))
    return it


# ------------------------------------------------------------------ recomendaciones
def recomendaciones(nv, s):
    r = []
    if s["vigor"] <= -0.5:
        r.append(("Revisar el establecimiento del cultivo", "El vigor de febrero a mayo está por debajo de su zona. Revisa fecha y densidad de siembra, humedad al sembrar, emergencia pareja y fertilización de arranque; si hubo fallas, considera resiembra o ajustar la fecha el próximo ciclo."))
    if s["humedad"] <= -0.5 or (not np.isnan(s["lluvia_pct"]) and s["lluvia_pct"] <= -10):
        r.append(("Cuidar la disponibilidad de agua", "Los indicadores de humedad o la lluvia de abril a junio quedaron por debajo de su zona. Si hay riego, programa uno de auxilio en las etapas críticas; si es temporal, conserva humedad con rastrojo y labranza de conservación y procura sembrar al inicio de las lluvias."))
    if s["hetero"] >= 0.5:
        r.append(("Atender la desigualdad dentro de la parcela", "Hay sectores con desarrollo distinto. Revisa nivelación y drenaje, fertilidad pareja y focos de plaga, enfermedad o maleza; los mapas de NDVI ayudan a localizar los sectores débiles y manejarlos por separado."))
    if (not np.isnan(s["pendiente"]) and s["pendiente"] >= 5) or s["pend_z"] >= 1:
        r.append(("Proteger el suelo en pendiente", "Con terreno inclinado conviene sembrar en curvas a nivel o en contorno y mantener cobertura para reducir erosión y escurrimiento de agua y fertilizante."))
    if nv == "Bajo" and not r:
        r.append(("Buscar la causa en el manejo", "Sus índices no muestran un problema claro frente a su zona. El rendimiento bajo probablemente responde a la zona o a prácticas que el modelo no ve: variedad, fecha de siembra, fertilización, control de maleza y enfermedades. Revísalas con el productor."))
    if nv == "Alto":
        r.append(("Mantener y documentar el manejo", "Registra variedad, fecha de siembra, fertilización y labores de esta parcela para replicarlas en otras de la zona, y verifica la cosecha real al final del ciclo."))
    if nv in ("Medio", "Bajo"):
        r.append(("Dar seguimiento durante el ciclo", "Compara cada mes el NDVI o EVI de la parcela con el promedio de su zona (pestaña de consulta por parcela). Una caída frente a la zona entre marzo y mayo es una señal temprana para actuar a tiempo."))
    r.append(("Registrar datos de manejo", "El modelo no conoce variedad, fecha de siembra, fertilización ni sanidad. Capturarlos ciclo con ciclo permitirá explicar por qué una parcela rinde distinto a sus vecinas y mejorar las estimaciones futuras."))
    return r[:6]


# ------------------------------------------------------------------ lectura de una parcela
def lectura_parcela(pid, rend, p10, p90, base, corr, area, municipio, estado, contrib, anom_row, raw_row, escala, ref,
                    real=None, mun_conocido=True):
    tiene_real = real is not None and not np.isnan(real)
    valor = real if tiene_real else rend                      # si ya hay cosecha registrada, la lectura se hace sobre ella
    nv = nivel(valor, ref); ver = VEREDICTO[nv]
    lo, hi = umbrales(ref)
    dif = valor - ref["media"]; dmun = valor - base
    s = senales(anom_row, raw_row, escala)
    up = [f"{k} ({v:+.3f} t/ha)" for k, v in contrib[contrib > 0].head(2).items()]
    dn = [f"{k} ({v:+.3f} t/ha)" for k, v in contrib[contrib < 0].head(2).items()]
    por_que = [f"<b>Punto de partida:</b> el promedio de {municipio} es {base:.2f} t/ha. La ubicación es lo que más pesa en la estimación."]
    if up: por_que.append("<b>Empujan hacia arriba:</b> " + "; ".join(up) + ".")
    if dn: por_que.append("<b>Empujan hacia abajo:</b> " + "; ".join(dn) + ".")
    por_que.append(f"<b>Ajuste total por las variables de la parcela:</b> {corr:+.2f} t/ha.")
    avisos = []
    if nivel(p10, ref) != nivel(p90, ref):
        avisos.append(f"El rango de 80 % ({p10:.2f} a {p90:.2f} t/ha) cruza categorías, así que la clasificación es orientativa.")
    if abs(corr) < 0.15:
        avisos.append(f"El ajuste por variables es de solo {corr:+.2f} t/ha: casi toda la estimación sale del promedio de {municipio}. Los índices de esta parcela no la separan claramente de las demás de su zona.")
    if not mun_conocido:
        avisos.append("El municipio no tiene parcelas en los datos; se usó el promedio del estado o el global como punto de partida, por lo que la estimación es menos confiable.")
    if real is not None and not np.isnan(real):
        d = rend - real
        if abs(d) <= 0.5: avisos.append(f"El rendimiento real registrado fue {real:.2f} t/ha: la estimación se acercó ({d:+.2f} t/ha).")
        else: avisos.append(f"El rendimiento real registrado fue {real:.2f} t/ha y la estimación fue {rend:.2f} ({d:+.2f} t/ha). Esa diferencia apunta a factores que el modelo no ve: manejo, variedad, fecha de siembra, plagas o enfermedades, o clima extremo a escala local.")
    return {
        "titulo": f"Parcela {pid}" if pid else "Parcela nueva", "nivel": nv,
        "sello": ("RENDIMIENTO " + ("REAL " if tiene_real else "") + ver).upper(),
        "resumen": (f"Rendimiento real {ver.lower()}: la parcela cosechó {real:.2f} t/ha, {abs(dif):.2f} t/ha {'por encima' if dif >= 0 else 'por debajo'} del promedio regional ({ref['media']:.2f} t/ha). El modelo estimaba {rend:.2f} t/ha."
                    if tiene_real else
                    f"Rendimiento {ver.lower()}: se espera cosechar {rend:.2f} t/ha, {abs(dif):.2f} t/ha {'por encima' if dif >= 0 else 'por debajo'} del promedio regional ({ref['media']:.2f} t/ha)."),
        "secciones": [
            {"titulo": "Qué significa este resultado", "items": [
                f"<b>Rendimiento esperado: {rend:.2f} t/ha.</b> Es la cosecha que se estima por hectárea al final del ciclo.",
                f"<b>Producción esperada: {rend * area:.1f} t</b> en {area:.1f} ha (rendimiento multiplicado por la superficie).",
                f"<b>Rango de 80 %: {p10:.2f} a {p90:.2f} t/ha.</b> En 8 de cada 10 casos como este, el rendimiento real queda dentro de ese rango."]},
            {"titulo": "Es bueno o es bajo", "texto":
                f"El promedio de las parcelas con rendimiento conocido es {ref['media']:.2f} t/ha. Con {valor:.2f} t/ha{' (rendimiento real)' if tiene_real else ''}, esta parcela queda {abs(dif):.2f} t/ha {'sobre' if dif >= 0 else 'bajo'} ese promedio y {abs(dmun):.2f} t/ha {'sobre' if dmun >= 0 else 'bajo'} el promedio de {municipio}. "
                f"Se clasifica como <b>{nv.lower()}</b> ({ver.lower()}). Criterio: alto si supera {hi:.2f} t/ha y bajo si no pasa de {lo:.2f} t/ha (medio desvío estándar alrededor del promedio)."},
            {"titulo": "Por qué sale este resultado", "items": por_que},
            {"titulo": "Señales de la parcela frente a su zona", "items": _items_senales(s) or ["No hay suficientes datos para comparar con su zona."]}],
        "recomendaciones": recomendaciones(nv, s), "avisos": avisos}


# ------------------------------------------------------------------ lectura del panorama (mapa y estados)
def lectura_mapa(vista, ref):
    if vista.empty: return None
    lo, hi = umbrales(ref)
    n = len(vista); m = vista.rendimiento_predicho.mean()
    niv = vista.rendimiento_predicho.map(lambda v: nivel(v, ref)).value_counts()
    a, b, c = (int(niv.get(k, 0)) for k in ("Alto", "Medio", "Bajo"))
    mun = vista.groupby("Municipio").agg(n=("ID_POLIGONO", "size"), rend=("rendimiento_predicho", "mean"), prod=("produccion_esperada_t", "sum"))
    mun = mun[mun.n >= 3].sort_values("rend")
    items = []
    if len(mun) >= 2:
        mejores = mun.tail(2).iloc[::-1]; peores = mun.head(2)
        items.append("<b>Mayor rendimiento esperado:</b> " + "; ".join(f"{i} ({r.rend:.2f} t/ha, {int(r.n)} parcelas)" for i, r in mejores.iterrows()) + ".")
        items.append("<b>Menor rendimiento esperado:</b> " + "; ".join(f"{i} ({r.rend:.2f} t/ha, {int(r.n)} parcelas)" for i, r in peores.iterrows()) + ".")
    tot = vista.produccion_esperada_t.sum()
    if tot > 0:
        top = vista.groupby("Municipio").produccion_esperada_t.sum().sort_values(ascending=False)
        items.append(f"<b>Concentración de la producción:</b> {top.index[0]} aporta el {top.iloc[0] / tot * 100:.0f} % de la producción esperada.")
    rec = []
    if len(mun) >= 2 and mun.rend.iloc[0] < lo:
        rec.append(("Priorizar la asistencia técnica", f"En {mun.index[0]} el rendimiento esperado ({mun.rend.iloc[0]:.2f} t/ha) está en el rango bajo. Es la zona donde una visita técnica, el análisis de suelo y la revisión de la fecha de siembra y la fertilización pueden rendir más."))
    if len(mun) >= 2 and mun.rend.iloc[-1] >= hi:
        rec.append(("Aprender de las zonas fuertes", f"En {mun.index[-1]} el rendimiento esperado es alto ({mun.rend.iloc[-1]:.2f} t/ha). Documentar sus prácticas de manejo permite replicarlas en zonas de menor rendimiento."))
    rec.append(("Revisar parcelas atípicas", "Las parcelas que se separan de su zona se detectan en la pestaña de consulta por parcela, comparando sus curvas de vegetación con el promedio de la zona."))
    return {"titulo": "Panorama de las parcelas seleccionadas", "nivel": None, "sello": None,
            "resumen": f"{n} parcelas con rendimiento esperado medio de {m:.2f} t/ha y producción total de {tot:,.0f} t.",
            "secciones": [
                {"titulo": "Qué muestra el mapa", "texto": f"Según el promedio regional ({ref['media']:.2f} t/ha), <b>{a} parcelas tienen rendimiento alto</b>, <b>{b} medio</b> y <b>{c} bajo</b>. El color de cada círculo indica el rendimiento esperado y su tamaño, la superficie."},
                {"titulo": "Dónde está lo mejor y lo más débil", "items": items or ["Se necesitan al menos 3 parcelas por municipio para compararlos."]}],
            "recomendaciones": rec, "avisos": []}


# ------------------------------------------------------------------ lectura del modelo
def lectura_modelo(met, pred, pesos, combos, ref):
    tr = pred[pred.conjunto == "ENTRENAMIENTO"].copy()
    err = (tr.rendimiento_predicho - tr.rendimiento_t_ha).abs()
    f = met["modelo_final"]["aleatoria"]; b = met["linea_base_municipio"]["aleatoria"]
    cob = ((tr.rendimiento_t_ha >= tr.p10) & (tr.rendimiento_t_ha <= tr.p90)).mean() * 100
    ct = met["correccion_tipica_t_ha"]
    grp = pesos.groupby("grupo").peso_pct.sum().sort_values(ascending=False)
    frase = {"Satelital: Sentinel-2": "los índices de Sentinel-2", "Satelital: Planet": "los índices de Planet", "Clima": "el clima",
             "Topografía y parcela": "la topografía y las características de la parcela"}
    ind = pesos.dropna(subset=["familia_indice"]).groupby("familia_indice").peso_pct.sum().sort_values(ascending=False)
    clima = pesos[pesos.grupo == "Clima"].head(1)
    mejora = (b["RMSE"] - f["RMSE"]) / b["RMSE"] * 100
    return {"titulo": "Qué dicen los resultados del modelo", "nivel": None, "sello": None,
            "resumen": f"En promedio, el modelo se equivoca {f['MAE']:.2f} t/ha (cerca de {f['MAE'] / ref['media'] * 100:.0f} % del rendimiento medio) al estimar parcelas que no vio.",
            "secciones": [
                {"titulo": "Qué mide cada indicador", "items": [
                    f"<b>MAE {f['MAE']:.2f} t/ha:</b> error promedio entre lo estimado y lo real.",
                    f"<b>RMSE {f['RMSE']:.2f} t/ha:</b> parecido al MAE, pero castiga más los errores grandes.",
                    f"<b>R² {f['R2']:.2f}:</b> el modelo explica alrededor del {f['R2'] * 100:.0f} % de las diferencias de rendimiento entre parcelas."]},
                {"titulo": "Qué tan útil es en la práctica", "items": [
                    f"El {(err <= 0.5).mean() * 100:.0f} % de las parcelas se estimó a menos de 0.5 t/ha del valor real y el {(err <= 1).mean() * 100:.0f} % a menos de 1 t/ha.",
                    f"El rango de 80 % contuvo el rendimiento real en el {cob:.0f} % de los casos, así que está bien calibrado.",
                    f"Casi todo el acierto viene de conocer el promedio de cada municipio: la corrección por variables es de ±{ct['desv_est']:.2f} t/ha y mejora el error en solo {mejora:.1f} % frente a usar únicamente ese promedio."]},
                {"titulo": "Qué variables pesan más", "items": [
                    f"Por tipo de variable, pesan más <b>{frase[grp.index[0]]}</b> ({grp.iloc[0]:.0f} % del peso total), seguidos de {frase[grp.index[1]]} ({grp.iloc[1]:.0f} %).",
                    f"El índice con más peso es <b>{ind.index[0]}</b> ({ind.iloc[0]:.0f} %). La variable de clima con más peso es <b>{clima.etiqueta.iloc[0]}</b>.",
                    "Un peso alto indica más influencia en la estimación, no necesariamente que sea la causa del rendimiento."]}],
            "recomendaciones": [
                ("Sumar más ciclos agrícolas", "Con un solo ciclo el modelo no puede aprender cómo cambia el rendimiento de un año a otro; varios ciclos darían más peso a los índices satelitales."),
                ("Incorporar datos de manejo", "Variedad, fecha de siembra, fertilización y sanidad explican mucho del rendimiento y hoy el modelo no los conoce."),
                ("Usar clima de mayor detalle", "El clima disponible tiene resolución de unos 5 km; datos de estaciones o de mayor resolución distinguirían mejor a parcelas vecinas."),
                ("Ampliar parcelas por municipio", "Con más parcelas por zona, el promedio local y las comparaciones entre vecinas serían más confiables.")],
            "avisos": ["La validación se hizo sobre 138 parcelas de un solo ciclo; los resultados pueden cambiar con más datos."]}


# ------------------------------------------------------------------ lectura del simulador
def lectura_simulador(p0, p1, mun0, mun1, base0, base1, ref):
    d, dm = p1 - p0, base1 - base0
    dv = d - dm
    items = [f"<b>Escenario:</b> {p1:.2f} t/ha frente a {p0:.2f} t/ha de la parcela original ({d:+.2f} t/ha)."]
    if mun0 != mun1:
        items.append(f"<b>Cambio de municipio</b> ({mun0} a {mun1}): el punto de partida pasa de {base0:.2f} a {base1:.2f} t/ha ({dm:+.2f} t/ha).")
    items.append(f"<b>Efecto de mover las variables:</b> {dv:+.2f} t/ha.")
    cierre = ("La ubicación pesa mucho más que cualquier variable de la parcela." if abs(dm) > abs(dv) * 3 and mun0 != mun1 else
              "Mover las variables cambia poco la estimación: dentro de una misma zona, ninguna variable por sí sola separa mucho a una parcela de otra.")
    return {"titulo": "Qué muestra este escenario", "nivel": nivel(p1, ref), "sello": f"ESCENARIO {VEREDICTO[nivel(p1, ref)]}".upper(),
            "resumen": cierre, "secciones": [{"titulo": "Comparación", "items": items}],
            "recomendaciones": [("Usar el simulador para explorar, no para decidir", "Los índices satelitales reflejan el estado del cultivo, pero no son palancas directas. Para mejorar el rendimiento hay que actuar sobre el manejo (siembra, fertilización, agua, sanidad) y dar seguimiento a cómo responden los índices.")],
            "avisos": []}


# ------------------------------------------------------------------ lectura de uso financiero
def lectura_financiera(base, umbral, orden):
    n = len(base)
    pct = base.Clasificación.value_counts(normalize=True).reindex(orden).fillna(0) * 100
    sup = base.groupby("Clasificación").area_ha.sum().reindex(orden).fillna(0)
    return {"titulo": "Qué significa esta clasificación", "nivel": None, "sello": None,
            "resumen": f"Con una referencia de {umbral:.1f} t/ha, el {pct[orden[0]]:.0f} % de las parcelas la supera con holgura, el {pct[orden[1]]:.0f} % es incierto y el {pct[orden[2]]:.0f} % está en riesgo.",
            "secciones": [
                {"titulo": "Cómo se clasifica", "items": [
                    "<b>Holgada:</b> aun en el caso pesimista del rango de 80 %, el rendimiento esperado supera la referencia.",
                    "<b>Incierta:</b> la referencia cae dentro del rango; la estimación no basta para asegurar si se cumple.",
                    "<b>En riesgo:</b> aun en el caso optimista del rango, el rendimiento queda por debajo de la referencia."]},
                {"titulo": "Superficie involucrada", "items": [f"{k.split(' (')[0]}: {sup[k]:,.0f} ha ({int((base.Clasificación == k).sum())} parcelas)." for k in orden]}],
            "recomendaciones": [
                ("Verificar en campo las parcelas inciertas", "Son las que más se benefician de una visita o de información adicional antes de tomar una decisión de financiamiento."),
                ("Ligar condiciones al riesgo", "Para las parcelas en riesgo se pueden considerar montos o plazos ajustados al rendimiento esperado, seguro agrícola o acompañamiento técnico."),
                ("Definir la referencia con costos reales", "La referencia debe salir del costo de producción y del precio de la cebada de la región, no solo de los datos del modelo.")],
            "avisos": ["Ejemplo ilustrativo basado en 138 parcelas y un solo ciclo. No sustituye un análisis de crédito: no considera costos, precios, manejo ni riesgos de mercado."]}


# ------------------------------------------------------------------ explicaciones de gráficas
EXPLICA = {
    "mapa": "Cada círculo es una parcela. El color indica el rendimiento esperado (terracota bajo, ámbar medio, verde oscuro alto) y el tamaño, su superficie. Sirve para ubicar rápido dónde se espera más y menos cosecha.",
    "estados": "Cada caja resume las parcelas de un estado: la línea central es la mediana, la caja contiene a la mitad de las parcelas y los puntos son parcelas individuales. Una caja más alta significa mayor rendimiento esperado.",
    "evi": "El EVI mide el vigor de la vegetación: valores altos indican un cultivo más desarrollado. La línea verde es la parcela y la gris punteada el promedio de su zona; si la parcela va por debajo entre marzo y mayo, su cultivo arrancó con menos vigor.",
    "ndvi": "El NDVI también mide el vigor de la vegetación, ahora con el satélite Sentinel-2. Compara la parcela (línea verde) con su zona (gris punteada) a lo largo del ciclo; la franja verde marca de abril a octubre.",
    "lluvia": "Lluvia mensual en el píxel de unos 5 km donde cae la parcela. Las lluvias de los primeros meses del ciclo suelen ser decisivas para el desarrollo del cultivo; parcelas vecinas comparten el mismo dato.",
    "temp": "Temperatura máxima y mínima mensual. Calor alto durante el desarrollo temprano puede acelerar el ciclo y reducir el rendimiento. Es un dato de unos 5 km, por lo que casi no distingue a parcelas vecinas.",
    "contrib": "Cada barra es una variable. Verde suma al rendimiento esperado frente al promedio de su municipio y rojo resta; la longitud indica cuánto, en t/ha. Las barras son cortas porque el efecto de cada variable es pequeño.",
    "pesos_grupo": "Reparte el peso total del modelo entre tipos de variables, después de llevarlas todas a la misma escala. Un grupo con más peso influye más en la corrección que el modelo hace sobre el promedio del municipio.",
    "pesos_indice": "Suma el peso de todos los meses de cada índice de vegetación. Muestra qué índice y qué satélite aportan más a la corrección.",
    "pesos_top": "Las 15 variables con más influencia. Una barra verde significa que valores más altos de esa variable (respecto a su zona) suben la estimación; una roja, que la bajan.",
    "combos": "Cada fila prueba el modelo con un conjunto distinto de variables. RMSE menor significa menor error. Si las diferencias son pequeñas, los grupos extra de variables casi no mejoran al promedio del municipio.",
    "finanzas": "Cada barra cuenta las parcelas de un estado según su rendimiento esperado frente a la referencia que elegiste, considerando el rango de 80 %.",
}


# ------------------------------------------------------------------ IA generativa (opcional)
SISTEMA_IA = ("Eres el asesor agronómico de SisAgro, un sistema que estima el rendimiento de cebada. Recibirás hechos calculados "
              "por el sistema. Redacta una lectura para un productor o técnico, en español claro y cálido, de 180 a 230 palabras, "
              "con tres secciones breves: «Qué significa», «Es bueno o es bajo» y «Qué hacer». Usa únicamente las cifras del "
              "contexto; no inventes datos ni afirmes causas que no estén respaldadas. Si algo es incierto, dilo. Las "
              "recomendaciones deben ser prácticas y cautelosas, y recuerda que no sustituyen a un técnico agrónomo.")


def a_texto(l):
    """Convierte una lectura (dict) en texto plano para pasarlo como contexto a la IA."""
    import re
    limpio = lambda s: re.sub(r"<[^>]+>", "", s)
    out = [l["titulo"], limpio(l.get("resumen", ""))]
    for s in l.get("secciones", []):
        out.append(s["titulo"] + ": " + limpio(s.get("texto", "")) + " " + " ".join(limpio(i) for i in s.get("items", [])))
    out.append("Recomendaciones del motor: " + " ".join(f"{r[0]}: {limpio(r[1])}" for r in l.get("recomendaciones", [])))
    out.append("Avisos: " + " ".join(limpio(a) for a in l.get("avisos", [])))
    return "\n".join(out)


def clave_contexto(texto):
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


def redactar_con_ia(contexto, api_key, modelo="claude-sonnet-5-5", timeout=40):
    """Devuelve (texto, None) o (None, mensaje de error). Nunca lanza excepciones."""
    try:
        import requests
        r = requests.post("https://api.anthropic.com/v1/messages",
                          headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": modelo, "max_tokens": 700, "system": SISTEMA_IA,
                                "messages": [{"role": "user", "content": contexto}]}, timeout=timeout)
        if r.status_code != 200:
            return None, f"El servicio de IA respondió con el código {r.status_code}."
        texto = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
        return (texto, None) if texto else (None, "La IA no devolvió texto.")
    except Exception as e:                                   # sin red, tiempo agotado, etc.
        return None, f"No se pudo contactar al servicio de IA ({type(e).__name__})."
