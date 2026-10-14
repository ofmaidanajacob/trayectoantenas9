import io

import folium
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from folium.plugins import Fullscreen, MeasureControl, TimestampedGeoJson
from streamlit_folium import st_folium

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Software de Movilidad Forense")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; }
        h1 { font-size: 22px !important; margin-top: -10px; }
        .metric-container {
            background-color: #f0f2f6; padding: 10px; border-radius: 8px;
            border: 1px solid #d1d5db; margin-bottom: 10px; text-align: center;
        }
        .metric-title { font-size: 18px; font-weight: bold; color: #1e3a8a; margin: 0; }
    </style>
""", unsafe_allow_html=True)

PALETA = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4", "#ec4899", "#84cc16"]

# Paso del reloj en modo reproducción: etiqueta -> (periodo ISO 8601, Timedelta)
PASOS = {
    "1 minuto": ("PT1M", pd.Timedelta(minutes=1)),
    "5 minutos": ("PT5M", pd.Timedelta(minutes=5)),
    "15 minutos": ("PT15M", pd.Timedelta(minutes=15)),
    "1 hora": ("PT1H", pd.Timedelta(hours=1)),
    "6 horas": ("PT6H", pd.Timedelta(hours=6)),
    "1 día": ("P1D", pd.Timedelta(days=1)),
}


# --- FUNCIONES GEODÉSICAS ---
def calculate_endpoint(lat, lon, bearing, d_km):
    """Coordenadas finales dado punto inicial, rumbo (grados) y distancia (km)."""
    R = 6371.0
    lat_rad, lon_rad, b = np.radians(lat), np.radians(lon), np.radians(bearing)
    d_r = d_km / R
    lat2 = np.arcsin(np.sin(lat_rad) * np.cos(d_r) + np.cos(lat_rad) * np.sin(d_r) * np.cos(b))
    lon2 = lon_rad + np.arctan2(
        np.sin(b) * np.sin(d_r) * np.cos(lat_rad),
        np.cos(d_r) - np.sin(lat_rad) * np.sin(lat2),
    )
    return np.degrees(lat2), np.degrees(lon2)


def draw_sector(m, lat, lon, azimuth, beamwidth, radius_km, color="#10b981"):
    """Sector de cobertura (azimuth + ancho de haz + radio)."""
    if pd.isna(azimuth) or pd.isna(beamwidth) or pd.isna(radius_km):
        return
    points = [[lat, lon]]
    for angle in np.linspace(azimuth - beamwidth / 2.0, azimuth + beamwidth / 2.0, 30):
        lat_p, lon_p = calculate_endpoint(lat, lon, angle, radius_km)
        points.append([lat_p, lon_p])
    points.append([lat, lon])
    folium.Polygon(
        locations=points, color=color, fill=True, fill_color=color,
        fill_opacity=0.30, weight=2, interactive=False,
    ).add_to(m)


# --- CARGA DE DATOS (con caché) ---
@st.cache_data(show_spinner="Procesando antenas...")
def cargar_antenas(data: bytes, forzar_sur_oeste: bool):
    df = pd.read_excel(io.BytesIO(data))
    df.columns = df.columns.str.strip()
    for col in ["Lat", "Long"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
            if forzar_sur_oeste:
                df[col] = df[col].apply(lambda x: -abs(x) if pd.notna(x) and x != 0 else x)
    df["CeldaID"] = df["CeldaID"].astype(str).str.strip()
    # Si una celda figura repetida, el merge duplicaría movimientos: nos quedamos con la primera
    return df.drop_duplicates(subset="CeldaID").reset_index(drop=True)


@st.cache_data(show_spinner="Procesando movimientos...")
def cargar_movimientos(data: bytes, nombre_archivo: str, df_ant: pd.DataFrame):
    df_mov = pd.read_excel(io.BytesIO(data))
    df_mov.columns = df_mov.columns.str.strip()
    nombre = str(df_mov.iloc[0, 0]) if not df_mov.empty else nombre_archivo
    df_mov["Celda"] = df_mov["Celda"].astype(str).str.strip()

    # LEFT merge para poder informar qué registros NO se lograron ubicar
    df = df_mov.merge(df_ant, left_on="Celda", right_on="CeldaID", how="left")
    sin_ubic = df.loc[df["Lat"].isna() | df["Long"].isna(), "Celda"].unique().tolist()

    df["Timestamp"] = pd.to_datetime(
        df["Fecha"].astype(str) + " " + df["Hora"].astype(str), dayfirst=True, errors="coerce"
    )
    sin_fecha = int(df["Timestamp"].isna().sum())
    df_ok = df.dropna(subset=["Lat", "Long", "Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df_ok, nombre, len(df_mov), sin_ubic, sin_fecha


# --- FUNCIONES DE MAPA ---
def crear_mapa(centro, tipo_mapa, zoom=14):
    if tipo_mapa == "Satelital":
        tiles, attr = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", "Google"
    else:
        tiles, attr = "OpenStreetMap", "OpenStreetMap"
    m = folium.Map(location=centro, zoom_start=zoom, tiles=tiles, attr=attr)
    Fullscreen().add_to(m)
    MeasureControl(primary_length_unit="kilometers").add_to(m)
    return m


def capas_comunes(m, df_ant, ver_todas, ver_antenas_periodo, activas):
    if ver_todas:
        for t in df_ant.dropna(subset=["Lat", "Long"]).itertuples():
            folium.CircleMarker(
                [t.Lat, t.Long], radius=3, color="#9ca3af", fill=True,
                tooltip=f"Celda: {t.CeldaID}",
            ).add_to(m)
    if ver_antenas_periodo:
        for l in activas:
            g = l["periodo"].groupby(["Celda", "Lat", "Long"]).size().reset_index(name="n")
            for r in g.itertuples():
                folium.CircleMarker(
                    [r.Lat, r.Long], radius=5, color=l["color"], fill=True, fill_opacity=0.5,
                    tooltip=f"{l['nombre']} · Celda {r.Celda} · {r.n} eventos",
                ).add_to(m)
    for wp in st.session_state.waypoints:
        folium.Marker(
            [wp["lat"], wp["lon"]], tooltip=wp["nombre"],
            icon=folium.Icon(color="blue", icon="location-dot", prefix="fa"),
        ).add_to(m)


def leyenda(m, activas):
    items = "".join(
        f"<div><span style='background:{l['color']};width:16px;height:4px;display:inline-block;"
        f"margin-right:6px;vertical-align:middle'></span>{l['nombre']}</div>"
        for l in activas
    )
    html = (
        "<div style='position:fixed;bottom:25px;left:25px;z-index:9999;background:white;"
        "padding:8px 10px;border:1px solid #999;border-radius:6px;font-size:12px'>"
        f"<b>Líneas</b>{items}</div>"
    )
    m.get_root().html.add_child(folium.Element(html))


# --- ESTADO ---
if "waypoints" not in st.session_state:
    st.session_state.waypoints = []

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("📂 Carga de Datos")
    file_antenas = st.file_uploader("Antenas (Excel)", type=["xlsx"])
    files_mov = st.file_uploader("Movimientos (Excel) — uno por línea", type=["xlsx"], accept_multiple_files=True)

    st.divider()
    st.header("⚙️ Visualización")
    tipo_mapa = st.radio("Capa Base:", ["Satelital", "Común"])
    ver_rastro = st.checkbox("Mostrar camino recorrido (modo evento)", value=True)
    ver_antenas_periodo = st.checkbox("Mostrar antenas del período", value=True)
    ver_todas = st.checkbox("Mostrar TODAS las torres del Excel", value=False)
    forzar_sur_oeste = st.checkbox("Forzar coordenadas Sur/Oeste (Argentina)", value=True)

    st.divider()
    st.header("📍 Waypoints Manuales")
    wp_nombre = st.text_input("Nombre / Referencia")
    wp_lat = st.text_input("Latitud")
    wp_lon = st.text_input("Longitud")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Añadir Punto", use_container_width=True) and wp_nombre and wp_lat and wp_lon:
            try:
                st.session_state.waypoints.append({
                    "nombre": wp_nombre,
                    "lat": float(wp_lat.replace(",", ".")),
                    "lon": float(wp_lon.replace(",", ".")),
                })
                st.rerun()
            except ValueError:
                st.error("Coordenadas inválidas")
    with c2:
        if st.session_state.waypoints and st.button("Limpiar Puntos", use_container_width=True):
            st.session_state.waypoints = []
            st.rerun()

# --- LÓGICA PRINCIPAL ---
st.title("🛡️ Sistema de Análisis y Movilidad Forense")

if not file_antenas or not files_mov:
    st.info("Cargá el Excel de antenas y uno o más Excel de movimientos (uno por línea).")
else:
    try:
        df_ant = cargar_antenas(file_antenas.getvalue(), forzar_sur_oeste)

        lineas = []
        for i, f in enumerate(files_mov):
            df, nombre, total, sin_ubic, sin_fecha = cargar_movimientos(f.getvalue(), f.name, df_ant)
            descartados = total - len(df)
            if descartados:
                with st.expander(f"⚠️ {nombre}: {descartados} de {total} registros descartados"):
                    st.write(f"Celdas sin ubicación en el Excel de antenas: {len(sin_ubic)}")
                    if sin_ubic:
                        st.code(", ".join(map(str, sin_ubic[:200])))
                    if sin_fecha:
                        st.write(f"Registros con fecha/hora inválida: {sin_fecha}")
            if not df.empty:
                lineas.append({"nombre": nombre, "df": df})

        # Color y visibilidad por línea
        with st.sidebar:
            st.divider()
            st.header("🎨 Líneas")
            for i, l in enumerate(lineas):
                cc, cn = st.columns([1, 3])
                l["color"] = cc.color_picker("c", PALETA[i % len(PALETA)], key=f"color_{i}", label_visibility="collapsed")
                l["visible"] = cn.checkbox(l["nombre"], value=True, key=f"vis_{i}")

        activas = [l for l in lineas if l["visible"]]
        if not activas:
            st.warning("No hay líneas activas con datos para mostrar.")
            st.stop()

        st.subheader("Investigación — Abonados: " + ", ".join(f"`{l['nombre']}`" for l in activas))

        # 1. FILTRO DE TIEMPO (global a todas las líneas)
        min_ts = min(l["df"]["Timestamp"].min() for l in activas)
        max_ts = max(l["df"]["Timestamp"].max() for l in activas)
        ct1, ct2 = st.columns(2)
        with ct1:
            d_i = st.date_input("Desde:", min_ts.date())
            t_i = st.time_input("Hora Inicio", min_ts.time())
            dt_inicio = pd.to_datetime(f"{d_i} {t_i}")
        with ct2:
            d_f = st.date_input("Hasta:", max_ts.date())
            t_f = st.time_input("Hora Fin", max_ts.time())
            dt_fin = pd.to_datetime(f"{d_f} {t_f}")

        for l in activas:
            d = l["df"]
            l["periodo"] = d[(d["Timestamp"] >= dt_inicio) & (d["Timestamp"] <= dt_fin)].reset_index(drop=True)
        activas = [l for l in activas if not l["periodo"].empty]
        if not activas:
            st.warning("No hay registros en el rango seleccionado.")
            st.stop()

        todos = pd.concat([l["periodo"] for l in activas])
        centro = [todos["Lat"].mean(), todos["Long"].mean()]
        bounds = [[todos["Lat"].min(), todos["Long"].min()], [todos["Lat"].max(), todos["Long"].max()]]

        tab_play, tab_manual, tab_datos = st.tabs(["▶️ Reproducción", "🎚️ Evento por evento", "📊 Datos y resumen"])

        # ================= TAB 1: REPRODUCCIÓN (play) =================
        with tab_play:
            ca, cb, cc_ = st.columns(3)
            paso_label = ca.selectbox("Paso del reloj", list(PASOS), index=3)
            ms = cb.slider("Velocidad (ms por paso)", 50, 2000, 300, step=50)
            auto = cc_.checkbox("Iniciar automáticamente", value=False)

            periodo_iso, paso_td = PASOS[paso_label]
            n_pasos = int((todos["Timestamp"].max() - todos["Timestamp"].min()) / paso_td) + 1
            if n_pasos > 3000:
                st.warning(f"Con paso de {paso_label} hay {n_pasos} pasos. Probá un paso mayor para una animación más fluida.")

            features = []
            for l in activas:
                d = l["periodo"]
                times = d["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
                coords = [[lon, lat] for lat, lon in zip(d["Lat"], d["Long"])]
                icon_style = {"fillColor": l["color"], "fillOpacity": 0.95, "stroke": "true",
                              "color": "white", "weight": 2, "radius": 8}
                if len(d) == 1:
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": coords[0]},
                        "properties": {"time": times[0], "icon": "circle", "iconstyle": icon_style,
                                       "popup": f"<b>{l['nombre']}</b><br>Celda {d.loc[0, 'Celda']}"},
                    })
                else:
                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "LineString", "coordinates": coords},
                        "properties": {"times": times, "icon": "circle", "iconstyle": icon_style,
                                       "style": {"color": l["color"], "weight": 4, "opacity": 0.8},
                                       "popup": f"<b>{l['nombre']}</b>"},
                    })

            m = crear_mapa(centro, tipo_mapa)
            capas_comunes(m, df_ant, ver_todas, ver_antenas_periodo, activas)
            TimestampedGeoJson(
                {"type": "FeatureCollection", "features": features},
                period=periodo_iso, add_last_point=True, auto_play=auto, loop=False,
                loop_button=True, max_speed=10, transition_time=ms,
                date_options="YYYY-MM-DD HH:mm:ss", time_slider_drag_update=True,
            ).add_to(m)
            leyenda(m, activas)
            m.fit_bounds(bounds)

            html_mapa = m.get_root().render()
            components.html(html_mapa, height=650)
            st.download_button("⬇️ Descargar mapa animado (HTML)", html_mapa.encode("utf-8"),
                               "mapa_animado.html", "text/html")
            st.caption("Los sectores de cobertura se muestran en la pestaña «Evento por evento». "
                       "Las horas del reloj coinciden con las del Excel.")

        # ================= TAB 2: EVENTO POR EVENTO =================
        with tab_manual:
            timeline = todos["Timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
            if len(timeline) > 1:
                idx = st.slider("Línea de tiempo (eventos):", 0, len(timeline) - 1, step=1, key="slider_eventos")
            else:
                idx = 0
            ts = timeline.iloc[idx]

            st.markdown(
                f"<div class='metric-container'><div class='metric-title'>📅 Evento {idx + 1} de {len(timeline)} — "
                f"{ts.strftime('%d/%m/%Y %H:%M:%S')}</div></div>",
                unsafe_allow_html=True,
            )

            actuales = []
            for l in activas:
                hasta = l["periodo"][l["periodo"]["Timestamp"] <= ts]
                if not hasta.empty:
                    actuales.append((l, hasta, hasta.iloc[-1]))

            centro_act = [np.mean([r["Lat"] for _, _, r in actuales]), np.mean([r["Long"] for _, _, r in actuales])]
            m2 = crear_mapa(centro_act, tipo_mapa)
            capas_comunes(m2, df_ant, ver_todas, ver_antenas_periodo, activas)

            filas = []
            for l, hasta, reg in actuales:
                if ver_rastro and len(hasta) > 1:
                    folium.PolyLine(hasta[["Lat", "Long"]].values.tolist(), color=l["color"], weight=3, opacity=0.8).add_to(m2)
                draw_sector(m2, reg["Lat"], reg["Long"], reg.get("Azimuth", 0), reg.get("A. Horiz", 60),
                            reg.get("Rad Cob (KM)", 1.0), color=l["color"])
                folium.CircleMarker(
                    [reg["Lat"], reg["Long"]], radius=9, color="white", weight=2, fill=True,
                    fill_color=l["color"], fill_opacity=1,
                    popup=f"<b>{l['nombre']}</b><br>Celda: {reg['Celda']}<br>Azimuth: {reg.get('Azimuth')}°",
                ).add_to(m2)
                filas.append({
                    "Línea": l["nombre"],
                    "Último evento": reg["Timestamp"].strftime("%d/%m/%Y %H:%M:%S"),
                    "Celda": str(reg["Celda"]),
                    "Coordenadas": f"{reg['Lat']:.5f}, {reg['Long']:.5f}",
                    "Azimuth": reg.get("Azimuth", "N/A"),
                    "Ancho": reg.get("A. Horiz", "N/A"),
                    "Radio (km)": reg.get("Rad Cob (KM)", "N/A"),
                })
            leyenda(m2, activas)
            st_folium(m2, height=520, key="mapa_forense", returned_objects=[], use_container_width=True)
            st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

        # ================= TAB 3: DATOS =================
        with tab_datos:
            todos_l = pd.concat([l["periodo"].assign(Linea=l["nombre"]) for l in activas]).sort_values("Timestamp")
            st.markdown("**Celdas utilizadas por línea** (cantidad de eventos, primer y último uso)")
            resumen = (
                todos_l.groupby(["Linea", "Celda"])
                .agg(Eventos=("Timestamp", "size"), Primero=("Timestamp", "min"), Ultimo=("Timestamp", "max"))
                .reset_index().sort_values(["Linea", "Eventos"], ascending=[True, False])
            )
            st.dataframe(resumen, use_container_width=True, hide_index=True)
            st.markdown("**Registros del período**")
            st.dataframe(todos_l, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Descargar registros (CSV)",
                               todos_l.to_csv(index=False, sep=";").encode("utf-8-sig"),
                               "registros_periodo.csv", "text/csv")

    except Exception as e:
        st.error(f"Error procesando los datos: {e}")

st.markdown(
    "<p style='text-align: center; color: #9ca3af; font-size: 11px; margin-top: 15px;'>"
    "Software Forense de Movilidad — Oficial Maidana Jacob</p>",
    unsafe_allow_html=True,
)
