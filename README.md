# SisAgro

Sistema de análisis y predicción de rendimiento de cebada para el Reto AgroCebada FIRA 2026
(Puebla, Hidalgo y Tlaxcala, ciclo abril-octubre 2025). Aplicación web en Streamlit y cuaderno de Colab.

## Ejecutar la aplicación
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar (enlace para el jurado)
1. Sube esta carpeta a un repositorio de GitHub. El archivo `.gitignore` ya excluye `venv/` y las claves.
2. En Streamlit Community Cloud crea una app nueva, elige el repositorio y `app.py` como archivo principal.

## Asesor SisAgro (interpretación de resultados)
Al final de cada análisis y gráfica, el asesor explica qué significa, si el rendimiento es bueno, regular o bajo
frente al promedio regional y qué hacer para mejorar.

- **Motor de interpretación (siempre activo):** reglas explícitas sobre las salidas del modelo (`asesor.py`). Todo número
  que muestra sale de los datos y del modelo; no necesita internet ni claves.
- **Lectura ampliada con IA generativa (opcional):** si configuras una clave de API, aparece una casilla en la barra
  lateral y un botón en cada lectura para que un modelo de lenguaje redacte una versión más completa usando solo los hechos
  del motor. Para activarla:
  - En Streamlit Cloud: Settings, Secrets, y agrega `ANTHROPIC_API_KEY = "tu-clave"` (opcional: `ANTHROPIC_MODEL`).
  - En local: crea `.streamlit/secrets.toml` con la misma línea.
  - Cada sesión está limitada a 12 lecturas. El uso de la API tiene costo; si no hay clave, la app funciona igual.
  - Para el reporte: el *prompt* de sistema está en `asesor.py` (`SISTEMA_IA`) y debe anexarse junto con los demás prompts.

## Cuaderno de Colab
`AgroCebada_2026_Modelo.ipynb`: súbelo a Colab, ejecútalo completo y, cuando lo pida, sube `tabla_variables_agrocebada.csv`.
Hace todo el flujo CRISP-DM y genera `predicciones_59_parcelas.csv`. Usa el mismo `modelo.py` que la aplicación.

## Contenido
| Archivo | Para qué sirve |
|---|---|
| `app.py` | Interfaz: panorama, consulta por parcela, predecir parcela nueva (captura o CSV), simulador, modelo y pesos, uso financiero |
| `asesor.py` | Asesor: interpretación de resultados y gráficas, recomendaciones y capa opcional de IA |
| `estilo.py` | Identidad visual: paleta, tipografía (Fraunces y Hanken Grotesk) y tema de las gráficas |
| `modelo.py` | Modelo: variables homogeneizadas (anomalías y z-score) y regresión Ridge cuyos coeficientes son los pesos |
| `entrenar_y_predecir.py` | Reentrena, valida y regenera predicciones, `metricas.json`, `pesos_variables.csv` y `combinaciones.csv` |
| `assets/` | Logotipo de SisAgro e ícono |
| `tabla_variables_agrocebada.csv` | Una fila por parcela: variables satelitales, climáticas, topográficas y de ubicación |
| `analisis/` | Scripts de preparación de datos (Fase 3) y de las primeras rondas de comparación de modelos |

## Metodología (CRISP-DM)
1. **Comprensión del negocio:** predecir el rendimiento de 59 parcelas con 138 parcelas de rendimiento conocido.
2. **Comprensión de los datos:** Sentinel-2, Planet, CHIRPS, CHIRTS-ERA5 y CEM-INEGI.
3. **Preparación y homogeneización:** resumen mensual por parcela; cada variable como diferencia respecto al promedio de su
   municipio (anomalía), imputación con la mediana y estandarización (z-score).
4. **Modelado:** estimación = promedio del municipio + corrección Ridge sobre 74 variables estandarizadas (pesos).
5. **Evaluación:** validación cruzada aleatoria (5x3) y agrupada por píxel climático; RMSE, MAE y R².
6. **Despliegue:** esta aplicación.

## Regenerar la tabla desde los datos originales
```bash
python analisis/fase3_preparacion.py <carpeta DATASET_RETO_AGRO_2026 descomprimida> tabla_variables_agrocebada.csv
python entrenar_y_predecir.py
```
(requiere además `geopandas`, `rasterio` y `pyproj`).

## Limitaciones
Un solo ciclo agrícola y 138 parcelas con rendimiento; el clima tiene resolución de unos 5 km (25 píxeles distintos);
la ubicación (municipio) explica la mayor parte del rendimiento y el efecto de cada variable dentro de su zona es pequeño.
