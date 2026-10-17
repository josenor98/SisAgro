"""Identidad visual de SisAgro: paleta, tipografía, estilos de la interfaz y tema de las gráficas."""
import base64
import html
import plotly.graph_objects as go
import plotly.io as pio

P = dict(bosque="#1B4332", cebada="#6F9A2F", trigo="#D9A441", tierra="#8B5E34", crema="#F7F4EA",
         crema2="#ECE7D6", linea="#DDD6BC", texto="#1F2A22", suave="#5C6A5F", arcilla="#B5483A", agua="#2F6F8F")
ESCALA = [[0, "#B5483A"], [0.35, "#D9A441"], [0.65, "#8DB04A"], [1, "#1B4332"]]     # bajo -> alto
COLOR_NIVEL = {"Alto": "#3F7D2E", "Medio": "#C58F1F", "Bajo": "#B5483A"}
FUENTE_TITULOS, FUENTE_TEXTO = "Fraunces", "Hanken Grotesk"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Hanken+Grotesk:wght@400;500;600;700&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, label, p, li, td, th {{
    font-family: '{FUENTE_TEXTO}', system-ui, 'Segoe UI', sans-serif; }}
.stApp {{ background: {P['crema']}; color: {P['texto']}; }}
h1, h2, h3, h4 {{ font-family: '{FUENTE_TITULOS}', Georgia, serif !important; color: {P['bosque']}; letter-spacing: -0.01em; }}
h2 {{ font-weight: 600; margin-top: .4rem; }}
.block-container {{ padding-top: 1.4rem; max-width: 1280px; }}
[data-testid="stToolbar"], footer {{ visibility: hidden; height: 0; }}

/* encabezado */
.hero {{ display:flex; align-items:center; gap:26px; padding:22px 30px; border-radius:18px; margin-bottom:18px;
    background: repeating-linear-gradient(115deg, rgba(255,255,255,.035) 0 14px, rgba(255,255,255,0) 14px 28px), linear-gradient(120deg, {P['bosque']} 0%, #245C43 100%);
    border-bottom: 5px solid {P['trigo']}; box-shadow: 0 6px 22px rgba(27,67,50,.22); flex-wrap: wrap; }}
.hero-logo {{ background:#fff; border-radius:16px; padding:6px; box-shadow:0 3px 10px rgba(0,0,0,.25); line-height:0; }}
.hero-logo img {{ width:104px; height:104px; border-radius:12px; }}
.hero-txt {{ flex:1; min-width:260px; }}
.hero-txt .marca {{ font-family:'{FUENTE_TITULOS}', Georgia, serif; font-size:3rem; font-weight:700; color:{P['crema']}; line-height:1; letter-spacing:-0.02em; margin:0; }}
.hero-txt .marca span {{ color:{P['trigo']}; }}
.hero-txt .lema {{ color:#CFE0C0; font-size:1.05rem; margin:.45rem 0 0 0; }}
.hero-chips {{ display:flex; gap:8px; flex-wrap:wrap; }}
.hero-chips span {{ background:rgba(247,244,234,.12); color:{P['crema']}; border:1px solid rgba(247,244,234,.28); padding:6px 13px; border-radius:999px; font-size:.86rem; font-weight:500; }}

/* pestañas */
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:2px solid {P['linea']}; }}
.stTabs [data-baseweb="tab"] {{ height:auto; padding:10px 18px; border-radius:10px 10px 0 0; font-weight:600; color:{P['suave']}; }}
.stTabs [data-baseweb="tab"] p {{ font-weight:600; font-size:.97rem; }}
.stTabs [aria-selected="true"] {{ background:{P['bosque']}; }}
.stTabs [aria-selected="true"] p {{ color:{P['crema']} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{P['trigo']}; height:3px; }}

/* tarjetas de métricas */
[data-testid="stMetric"] {{ background:#fff; border:1px solid {P['linea']}; border-left:5px solid {P['trigo']}; border-radius:12px; padding:14px 16px; box-shadow:0 1px 4px rgba(27,67,50,.07); }}
[data-testid="stMetricLabel"] p {{ color:{P['suave']}; font-weight:600; font-size:.85rem; text-transform:uppercase; letter-spacing:.04em; }}
[data-testid="stMetricValue"] {{ font-family:'{FUENTE_TITULOS}', Georgia, serif; color:{P['bosque']}; font-weight:600; }}

/* botones */
.stButton > button, .stDownloadButton > button {{ border-radius:10px; font-weight:600; border:1px solid {P['bosque']}; color:{P['bosque']}; background:#fff; }}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {{ background:{P['bosque']}; color:{P['crema']}; border:1px solid {P['bosque']}; }}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {{ background:{P['trigo']}; border-color:{P['trigo']}; color:{P['bosque']}; }}

/* barra lateral */
[data-testid="stSidebar"] {{ background:{P['crema2']}; border-right:1px solid {P['linea']}; }}
[data-testid="stSidebar"] h2 {{ font-size:1.25rem; }}
.side-marca {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
.side-marca img {{ width:46px; height:46px; border-radius:10px; background:#fff; padding:2px; }}
.side-marca b {{ font-family:'{FUENTE_TITULOS}', Georgia, serif; font-size:1.5rem; color:{P['bosque']}; }}

/* cómo leer una gráfica */
.leer {{ background:#fff; border:1px dashed {P['linea']}; border-radius:10px; padding:10px 14px; margin:2px 0 14px 0; color:{P['suave']}; font-size:.9rem; }}
.leer b {{ color:{P['bosque']}; }}

/* asesor */
.asesor {{ background:#fff; border:1px solid {P['linea']}; border-radius:16px; margin:14px 0 8px 0; overflow:hidden; box-shadow:0 2px 10px rgba(27,67,50,.08); }}
.asesor-cab {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:13px 20px; background:{P['bosque']}; color:{P['crema']}; flex-wrap:wrap; }}
.asesor-cab .tit {{ font-family:'{FUENTE_TITULOS}', Georgia, serif; font-size:1.2rem; font-weight:600; }}
.asesor-cab .sub {{ font-size:.82rem; color:#CFE0C0; }}
.sello {{ padding:5px 14px; border-radius:999px; color:#fff; font-weight:700; font-size:.85rem; letter-spacing:.03em; }}
.asesor-cuerpo {{ padding:16px 22px 8px 22px; }}
.asesor-cuerpo h4 {{ font-size:1.02rem; margin:14px 0 6px 0; padding-bottom:4px; border-bottom:2px solid {P['crema2']}; }}
.asesor-cuerpo h4:first-child {{ margin-top:2px; }}
.asesor-cuerpo p {{ margin:.2rem 0 .5rem 0; line-height:1.5; }}
.asesor-cuerpo ul {{ margin:.2rem 0 .6rem 0; padding-left:1.15rem; }}
.asesor-cuerpo li {{ margin-bottom:.4rem; line-height:1.45; }}
.reco {{ border-left:4px solid {P['cebada']}; background:#F6F9EE; padding:8px 12px; border-radius:0 8px 8px 0; margin:6px 0; }}
.reco b {{ color:{P['bosque']}; }}
.aviso {{ border-left:4px solid {P['trigo']}; background:#FFF8E6; padding:8px 12px; border-radius:0 8px 8px 0; margin:6px 0; font-size:.93rem; }}
.pie-asesor {{ padding:8px 22px 14px 22px; color:{P['suave']}; font-size:.82rem; }}
"""


def encabezado(logo_png: bytes, parcelas: int, estados: int) -> str:
    b64 = base64.b64encode(logo_png).decode()
    return f"""<div class="hero">
  <div class="hero-logo"><img src="data:image/png;base64,{b64}" alt="Logotipo de SisAgro"></div>
  <div class="hero-txt"><p class="marca">Sis<span>Agro</span></p>
  <p class="lema">Sistema de análisis y predicción de rendimiento de cebada</p></div>
  <div class="hero-chips"><span>{parcelas} parcelas</span><span>{estados} estados</span><span>Ciclo abril a octubre 2025</span></div>
</div>"""


def marca_lateral(logo_png: bytes) -> str:
    b64 = base64.b64encode(logo_png).decode()
    return f'<div class="side-marca"><img src="data:image/png;base64,{b64}" alt=""><b>SisAgro</b></div>'


def como_leer(texto: str) -> str:
    return f'<div class="leer"><b>Cómo leer esta gráfica.</b> {html.escape(texto)}</div>'


def _lista(items):
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def bloque_asesor(l: dict) -> str:
    """Renderiza la lectura del asesor (dict generado en asesor.py) como tarjeta HTML."""
    sello = ""
    if l.get("sello"):
        color = COLOR_NIVEL.get(l.get("nivel"), P["suave"])
        sello = f'<span class="sello" style="background:{color}">{html.escape(l["sello"])}</span>'
    partes = [f'<div class="asesor"><div class="asesor-cab"><div><div class="tit">{html.escape(l["titulo"])}</div>'
              f'<div class="sub">Lectura del asesor SisAgro</div></div>{sello}</div><div class="asesor-cuerpo">']
    if l.get("resumen"):
        partes.append(f'<p style="font-size:1.04rem"><b>{l["resumen"]}</b></p>')
    for sec in l.get("secciones", []):
        partes.append(f'<h4>{html.escape(sec["titulo"])}</h4>')
        if sec.get("texto"):
            partes.append(f'<p>{sec["texto"]}</p>')
        if sec.get("items"):
            partes.append(_lista(sec["items"]))
    if l.get("recomendaciones"):
        partes.append("<h4>Qué hacer para mejorar</h4>")
        partes += [f'<div class="reco"><b>{r[0]}.</b> {r[1]}</div>' for r in l["recomendaciones"]]
    for a in l.get("avisos", []):
        partes.append(f'<div class="aviso">{a}</div>')
    partes.append('</div><div class="pie-asesor">Interpretación automática a partir de los resultados del modelo. '
                  'Es una orientación general y no sustituye la valoración de un técnico agrónomo.</div></div>')
    return "".join(partes)


def bloque_ia_html(texto: str) -> str:
    """Convierte el texto de la IA (párrafos, **negritas** y viñetas simples) en una tarjeta HTML segura."""
    import re
    partes, lista = [], []
    def cerrar():
        if lista:
            partes.append("<ul>" + "".join(f"<li>{i}</li>" for i in lista) + "</ul>"); lista.clear()
    fmt = lambda x: re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(x.strip()))
    for linea in texto.splitlines():
        if not linea.strip(): cerrar(); continue
        l = linea.strip().lstrip("#").strip() if linea.strip().startswith("#") else linea.strip()
        if l[:2] in ("- ", "* ", "• "): lista.append(fmt(l[2:]))
        else:
            cerrar(); partes.append(f"<p>{fmt(l)}</p>")
    cerrar()
    return ('<div class="asesor"><div class="asesor-cab"><div><div class="tit">Lectura ampliada con IA generativa</div>'
            '<div class="sub">Redactada por un modelo de lenguaje</div></div></div><div class="asesor-cuerpo">' + "".join(partes) +
            '</div><div class="pie-asesor">Texto generado a partir de los resultados del sistema. Revísalo antes de usarlo; '
            'no sustituye la valoración de un técnico agrónomo.</div></div>')


def registrar_tema():
    pio.templates["sisagro"] = go.layout.Template(layout=dict(
        font=dict(family=f"{FUENTE_TEXTO}, system-ui, sans-serif", color=P["texto"], size=13),
        title=dict(font=dict(family=f"{FUENTE_TITULOS}, Georgia, serif", size=17, color=P["bosque"])),
        colorway=[P["bosque"], P["trigo"], P["agua"], P["cebada"], P["tierra"], P["arcilla"]],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.0)",
        xaxis=dict(gridcolor="#E6E0C8", linecolor=P["linea"], zerolinecolor=P["linea"]),
        yaxis=dict(gridcolor="#E6E0C8", linecolor=P["linea"], zerolinecolor=P["linea"]),
        legend=dict(bgcolor="rgba(0,0,0,0)")))
    pio.templates.default = "sisagro"
