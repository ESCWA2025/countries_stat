# sheet_explorer_v2.py
# Run: streamlit run sheet_explorer_v2.py

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ------------------------------
# Minimal page chrome
# ------------------------------
st.set_page_config(
    page_title="Sheet Explorer V2",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] { display: none !important; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Helpers
# ------------------------------
EXPECTED_COLS = ["SRC_Country", "City", "Latitude", "Longitude", "count"]
COL_ALIASES = {
    "src_country": "SRC_Country",
    "country": "SRC_Country",
    "src country": "SRC_Country",
    "city": "City",
    "latitude": "Latitude",
    "lat": "Latitude",
    "longitude": "Longitude",
    "lon": "Longitude",
    "long": "Longitude",
    "count": "count",
    "n": "count",
    "freq": "count",
    "frequency": "count",
}

def color_for_sheet(name: str) -> list[int]:
    h = hashlib.md5(name.encode("utf-8")).digest()
    return [int(h[0]), int(h[1]), int(h[2]), 200]

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = re.sub(r"\s+", " ", str(c)).strip().lower()
        mapping[c] = COL_ALIASES.get(key, c)
    return df.rename(columns=mapping)

def _parse_int_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(s):
        return s
    if pd.api.types.is_float_dtype(s):
        return s.fillna(0).astype(int)
    s = s.astype(str).str.replace(r"[^\d]", "", regex=True)
    s = s.replace("", np.nan)
    return s.fillna(0).astype(int)

def _safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _strip_weird_whitespace(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def _clean_punctuation(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"^[\"':,\s]+", "", regex=True)
        .str.replace(r"[\"':,\s]+$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def _normcase(x: str) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip().casefold()

def _scale_radius(counts: pd.Series, min_r: int = 800, max_r: int = 7000, mode: str = "sqrt") -> np.ndarray:
    s = counts.astype(float).clip(lower=0)
    if s.nunique() <= 1:
        return np.full(len(s), (min_r + max_r) / 2.0)
    s_min, s_max = s.min(), s.max()
    rng = (s_max - s_min) if (s_max - s_min) != 0 else 1.0
    if mode == "linear":
        norm = (s - s_min) / rng
    elif mode == "log":
        s1 = np.log1p(s)
        nmin, nmax = s1.min(), s1.max()
        nrng = (nmax - nmin) if (nmax - nmin) != 0 else 1.0
        norm = (s1 - nmin) / nrng
    else:  # sqrt
        norm = np.sqrt((s - s_min) / rng)
    return (min_r + norm * (max_r - min_r)).to_numpy()

def _haversine_km_vec(lat_arr, lon_arr, lat0, lon0) -> np.ndarray:
    """Vectorized Haversine distance (km) from (lat_arr, lon_arr) to (lat0, lon0)."""
    R = 6371.0088
    lat1 = np.radians(lat_arr.astype(float))
    lon1 = np.radians(lon_arr.astype(float))
    lat2 = np.radians(float(lat0))
    lon2 = np.radians(float(lon0))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ------------------------------
# Load from /data
# ------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
EXCEL_PATH = DATA_DIR / "grouped_country.xlsx"
CITIES_JSON_PATH = DATA_DIR / "main_cities_with_radius.json"

@st.cache_data(show_spinner=False)
def read_excel_all_sheets_from_path(path: Path) -> Dict[str, pd.DataFrame]:
    dfs = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    cleaned = {}
    for sheet, df in dfs.items():
        df = _canonicalize_columns(df)
        if any(c not in df.columns for c in EXPECTED_COLS):
            continue
        df["SRC_Country"] = _clean_punctuation(_strip_weird_whitespace(df["SRC_Country"]))
        df["City"] = _clean_punctuation(_strip_weird_whitespace(df["City"]))
        df["Latitude"] = _safe_to_numeric(df["Latitude"])
        df["Longitude"] = _safe_to_numeric(df["Longitude"])
        df["count"] = _parse_int_series(df["count"])
        df = df.dropna(subset=["Latitude", "Longitude"])
        df["Sheet"] = sheet
        cleaned[sheet] = df
    return cleaned

@st.cache_data(show_spinner=False)
def load_main_cities_with_radius(path: Path) -> Dict[str, List[Dict[str, float]]]:
    """
    Returns: { country: [ { "name": str, "radius_km": number }, ... ], ... }
    """
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8-sig")
    raw = json.loads(text)
    out: Dict[str, List[Dict[str, float]]] = {}
    for country, items in raw.items():
        if not isinstance(country, str) or not isinstance(items, list):
            continue
        cleaned_items = []
        seen = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            rad = it.get("radius_km")
            if not isinstance(name, str):
                continue
            try:
                rad_val = float(rad)
            except Exception:
                continue
            key = _normcase(name)
            if key in seen:
                continue
            seen.add(key)
            cleaned_items.append({"name": name.strip(), "radius_km": rad_val})
        if cleaned_items:
            out[country] = cleaned_items
    return out

# Guards
if not EXCEL_PATH.exists():
    st.error(f"Missing Excel file: {EXCEL_PATH}")
    st.stop()
dfs = read_excel_all_sheets_from_path(EXCEL_PATH)
if not dfs:
    st.error("No valid sheets found in the Excel file.")
    st.stop()
main_cities_map = load_main_cities_with_radius(CITIES_JSON_PATH)

# ------------------------------
# UI (no headings): Country + City + marker-size controls
# ------------------------------
if main_cities_map:
    available_sheets = sorted(set(dfs.keys()).intersection(main_cities_map.keys())) or sorted(dfs.keys())
else:
    available_sheets = sorted(dfs.keys())

country_options = ["All"] + available_sheets

c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
with c1:
    sel_country = st.selectbox("Country", options=country_options, index=0, key="v2_country")

# City choices:
# - If a specific country: list cities from main_cities_with_radius.json (plus "All").
# - If "All": city filter stays "All" (no cross-country selection).
if sel_country != "All":
    city_items = main_cities_map.get(sel_country, [])
    city_choices = ["All"] + [it["name"] for it in city_items]
else:
    city_choices = ["All"]

with c2:
    sel_city = st.selectbox("City", options=city_choices, index=0, key="v2_city")

with c3:
    size_mode = st.radio("Bubble size scale", ["sqrt", "linear", "log"], index=0, horizontal=True, key="scale_mode")

min_r, max_r = st.columns(2)
with min_r:
    rmin = st.slider("Min bubble radius (m)", 100, 10000, 800, key="min_r_v2")
with max_r:
    rmax = st.slider("Max bubble radius (m)", 1000, 30000, 7000, key="max_r_v2")
if rmax <= rmin:
    rmax = rmin + 1

# ------------------------------
# Build dataset for map
# ------------------------------
if sel_country == "All":
    df_base = pd.concat(dfs.values(), ignore_index=True)
else:
    df_base = dfs[sel_country].copy()

df_base["__City_norm__"] = df_base["City"].apply(_normcase)

def _get_city_radius(country: str, city_name: str) -> Tuple[float, str]:
    """Return (radius_km, display_name) for city from JSON (case-insensitive)."""
    items = main_cities_map.get(country, [])
    lookup = { _normcase(it["name"]): it["radius_km"] for it in items }
    rn = _normcase(city_name)
    if rn in lookup:
        return float(lookup[rn]), [it["name"] for it in items if _normcase(it["name"]) == rn][0]
    return None, None

# If a specific city is selected (and a specific country), filter by its JSON radius
if sel_country != "All" and sel_city != "All":
    radius_km, disp_name = _get_city_radius(sel_country, sel_city)
    if radius_km is None:
        st.warning("Selected city not found in main_cities_with_radius.json for this country.")
        st.stop()

    # Determine the center (lat0, lon0) from rows that exactly match the city name (case-insensitive)
    city_rows = df_base[df_base["__City_norm__"] == _normcase(sel_city)]
    if city_rows.empty:
        st.warning("No rows in the data match the selected city name. Cannot compute center for radius filter.")
        st.stop()

    # Weighted-by-count center for the selected city
    if city_rows["count"].sum() > 0:
        lat0 = float(np.average(city_rows["Latitude"], weights=city_rows["count"]))
        lon0 = float(np.average(city_rows["Longitude"], weights=city_rows["count"]))
    else:
        lat0 = float(city_rows["Latitude"].mean())
        lon0 = float(city_rows["Longitude"].mean())

    # Compute distances for all rows in the selected country and filter within radius
    dist = _haversine_km_vec(df_base["Latitude"].to_numpy(), df_base["Longitude"].to_numpy(), lat0, lon0)
    df_base = df_base.assign(__dist_km__=dist)
    df_near = df_base[df_base["__dist_km__"] <= radius_km].copy()

    st.caption(f"Showing posts within {int(radius_km)} km of “{disp_name}”. Center at ({lat0:.4f}, {lon0:.4f}).")

    # Aggregate for map
    map_df = (
        df_near.groupby(["Sheet", "City", "Latitude", "Longitude"], as_index=False)
        .agg(count=("count", "sum"), distance_km=("__dist_km__", "min"))
        .rename(columns={"Latitude": "lat", "Longitude": "lon"})
        .sort_values(["distance_km", "count"], ascending=[True, False])
    )

    # Marker sizing & coloring
    map_df["radius"] = _scale_radius(map_df["count"], rmin, rmax, size_mode)
    map_df["color"] = map_df["Sheet"].apply(color_for_sheet)

    # View centered on the selected city
    if radius_km <= 10:
        zoom = 10
    elif radius_km <= 25:
        zoom = 9
    elif radius_km <= 50:
        zoom = 8
    elif radius_km <= 100:
        zoom = 7
    elif radius_km <= 200:
        zoom = 6
    elif radius_km <= 350:
        zoom = 5
    else:
        zoom = 4

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        opacity=0.6,
        auto_highlight=True,
    )
    view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom)

    MAP_STYLE = None
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "{Sheet} • {City}\n{count} posts\n{distance_km} km from center"},
            height=520,
            map_style=MAP_STYLE,
        ),
        use_container_width=True,
        key="v2_map_radius",
    )

    st.dataframe(
        map_df.rename(columns={"lat": "Latitude", "lon": "Longitude"}),
        use_container_width=True,
        hide_index=True
    )

else:
    # Country="All" OR City="All": standard (no radius filter) aggregation
    map_df = (
        df_base.groupby(["Sheet", "City", "Latitude", "Longitude"], as_index=False)["count"]
        .sum()
        .rename(columns={"Latitude": "lat", "Longitude": "lon"})
        .sort_values("count", ascending=False)
    )

    if map_df.empty:
        st.info("No geocoded rows to plot.")
        st.stop()

    map_df["radius"] = _scale_radius(map_df["count"], rmin, rmax, size_mode)
    map_df["color"] = map_df["Sheet"].apply(color_for_sheet)

    # View state
    if sel_country == "All":
        vlat = float(map_df["lat"].mean())
        vlon = float(map_df["lon"].mean())
        zoom = 3
    else:
        vlat = float(map_df["lat"].mean())
        vlon = float(map_df["lon"].mean())
        zoom = 4

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        opacity=0.6,
        auto_highlight=True,
    )
    view = pdk.ViewState(latitude=vlat, longitude=vlon, zoom=zoom)

    MAP_STYLE = None
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip={"text": "{Sheet} • {City}\n{count} posts"},
            height=520,
            map_style=MAP_STYLE,
        ),
        use_container_width=True,
        key="v2_map_all",
    )

    st.dataframe(
        map_df.rename(columns={"lat": "Latitude", "lon": "Longitude"}),
        use_container_width=True,
        hide_index=True
    )
