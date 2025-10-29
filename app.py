# job_location_dashboard.py
# Streamlit app for analyzing multi-sheet Excel of job vacancy locations.
# Each sheet corresponds to a country (plus an 'Unknown_Country' sheet).
# Expected columns per sheet: SRC_Country | City | Latitude | Longitude | count
#
# Run: streamlit run job_location_dashboard.py
# -----------------------------------------------------------------------------

import math
import re
from typing import Dict, List, Tuple, Optional
import hashlib
import json

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(
    page_title="Job Locations — Multi-Sheet Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Utilities
# ------------------------------

EXPECTED_COLS = ["SRC_Country", "City", "Latitude", "Longitude", "count"]
COL_ALIASES = {
    # lowercase stripped -> canonical
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
    # stable per-name color (same across runs)
    h = hashlib.md5(name.encode("utf-8")).digest()
    return [int(h[0]), int(h[1]), int(h[2]), 200]  # RGBA 0-255

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename close column variants to the expected canonical names."""
    mapping = {}
    for c in df.columns:
        key = re.sub(r"\s+", " ", str(c)).strip().lower()
        mapping[c] = COL_ALIASES.get(key, c)
    out = df.rename(columns=mapping)
    return out

def _parse_int_series(s: pd.Series) -> pd.Series:
    """
    Convert strings like '369,188' -> 369188, handle None/NaN, keep integers.
    Non-digits removed; empty becomes 0.
    """
    if pd.api.types.is_integer_dtype(s):
        return s
    if pd.api.types.is_float_dtype(s):
        return s.fillna(0).astype(int)

    s = s.astype(str).str.replace(r"[^\d]", "", regex=True)
    s = s.replace("", np.nan)
    return s.fillna(0).astype(int)

def _safe_to_numeric(s: pd.Series) -> pd.Series:
    """Coerce to float for lat/lon, safely handling bad values."""
    return pd.to_numeric(s, errors="coerce")

def _has_arabic(text: str) -> bool:
    if not isinstance(text, str):
        return False
    # Arabic Unicode block rough check
    return bool(re.search(r"[\u0600-\u06FF]", text))

def _camel_splits(s: str) -> Optional[str]:
    """
    Insert spaces into CamelCase or PascalCase like 'LebanonBeirutCity' -> 'Lebanon Beirut City'.
    Returns None if no change.
    """
    if not isinstance(s, str):
        return None
    out = re.sub(r"(?<=[A-Za-z])(?=[A-Z][a-z])", " ", s)
    return out if out != s else None

def _strip_weird_whitespace(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def _clean_punctuation(s: pd.Series) -> pd.Series:
    # Common weird prefixes/suffixes seen in examples: colons, quotes, trailing commas
    return (
        s.astype(str)
        .str.replace(r"^[\"':,\s]+", "", regex=True)
        .str.replace(r"[\"':,\s]+$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def _normcase(x: str) -> str:
    """Whitespace-trim + casefold for robust equality/lookup."""
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip().casefold()

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distance in KM between two lat/lon points."""
    R = 6371.0088  # mean earth radius
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

@st.cache_data(show_spinner=False)
def read_excel_all_sheets(file) -> Dict[str, pd.DataFrame]:
    """
    Read every sheet into a dict of DataFrames.
    - We do not assume sheet names; they represent countries + Unknown_Country.
    - We standardize column names and dtypes.
    """
    try:
        dfs = pd.read_excel(file, sheet_name=None, engine="openpyxl")
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        return {}

    cleaned = {}
    for sheet, df in dfs.items():
        df = _canonicalize_columns(df)

        # Only keep the expected columns that exist; warn if missing.
        missing = [c for c in EXPECTED_COLS if c not in df.columns]
        if missing:
            st.warning(f"Sheet '{sheet}' is missing columns: {missing}. It will be skipped.")
            continue

        # Clean and coerce
        df["SRC_Country"] = _clean_punctuation(_strip_weird_whitespace(df["SRC_Country"]))
        df["City"] = _clean_punctuation(_strip_weird_whitespace(df["City"]))
        df["Latitude"] = _safe_to_numeric(df["Latitude"])
        df["Longitude"] = _safe_to_numeric(df["Longitude"])
        df["count"] = _parse_int_series(df["count"])

        # Drop rows with missing lat/lon
        df = df.dropna(subset=["Latitude", "Longitude"])

        # Enrich
        df["is_arabic_country"] = df["SRC_Country"].apply(_has_arabic)
        df["is_arabic_city"] = df["City"].apply(_has_arabic)
        df["camel_split_city"] = df["City"].apply(_camel_splits)
        df["sheet_name"] = sheet

        cleaned[sheet] = df

    return cleaned

@st.cache_data(show_spinner=False)
def aggregated_overview(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate totals by sheet (country)."""
    rows = []
    for sheet, df in dfs.items():
        total_posts = int(df["count"].sum())
        n_unique_cities = int(df["City"].nunique())
        n_rows = int(len(df))
        pct_ar_city = float(df["is_arabic_city"].mean() * 100) if n_rows else 0.0
        pct_ar_country = float(df["is_arabic_country"].mean() * 100) if n_rows else 0.0
        rows.append({
            "Sheet": sheet,
            "Total posts": total_posts,
            "Rows": n_rows,
            "Unique cities": n_unique_cities,
            "% Arabic in City": round(pct_ar_city, 2),
            "% Arabic in SRC_Country": round(pct_ar_country, 2),
        })
    if not rows:
        return pd.DataFrame(columns=["Sheet", "Total posts", "Rows", "Unique cities", "% Arabic in City", "% Arabic in SRC_Country"])
    out = pd.DataFrame(rows).sort_values("Total posts", ascending=False, ignore_index=True)
    return out

def get_all_data(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all sheets with a 'Sheet' column."""
    if not dfs:
        return pd.DataFrame(columns=EXPECTED_COLS + ["is_arabic_country", "is_arabic_city", "camel_split_city", "sheet_name"])
    all_df = pd.concat(dfs.values(), ignore_index=True)
    all_df = all_df.rename(columns={"sheet_name": "Sheet"})
    return all_df

@st.cache_data(show_spinner=False)
def country_centroids(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute per-sheet weighted centroid using counts as weights."""
    rows = []
    for sheet, df in dfs.items():
        if df.empty:
            continue
        w = df["count"].clip(lower=0).replace([np.inf, -np.inf], 0).fillna(0).astype(float)
        lat = np.average(df["Latitude"], weights=w) if w.sum() > 0 else df["Latitude"].mean()
        lon = np.average(df["Longitude"], weights=w) if w.sum() > 0 else df["Longitude"].mean()
        rows.append({"Sheet": sheet, "centroid_lat": float(lat), "centroid_lon": float(lon), "Total posts": int(df["count"].sum())})
    return pd.DataFrame(rows)

def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0

def _scale_radius(counts: pd.Series, min_r: int = 1000, max_r: int = 8000, mode: str = "sqrt") -> np.ndarray:
    """
    Map counts -> radii (meters). Supports 'linear', 'sqrt', 'log'.
    Handles identical counts safely.
    """
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
    else:  # "sqrt"
        norm = np.sqrt((s - s_min) / rng)

    return (min_r + norm * (max_r - min_r)).to_numpy()

def _haversine_km_vec(lat_arr, lon_arr, lat0, lon0) -> np.ndarray:
    """Vectorized Haversine distance from arrays (lat_arr, lon_arr) to a single point (lat0, lon0)."""
    R = 6371.0088
    lat1 = np.radians(lat_arr.astype(float))
    lon1 = np.radians(lon_arr.astype(float))
    lat2 = np.radians(float(lat0))
    lon2 = np.radians(float(lon0))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) ** 2 * np.sin(dlon / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ------------------------------
# NEW: Main cities JSON helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def _load_main_cities_json_from_bytes(b: bytes) -> Dict[str, List[str]]:
    """
    Parse main_cities_lists.json bytes -> { country: [city1, city2, ...] }
    Cleans whitespace and deduplicates.
    """
    if not b:
        return {}
    try:
        # Handle BOM and ensure utf-8
        text = b.decode("utf-8-sig")
        raw = json.loads(text)
        if not isinstance(raw, dict):
            st.warning("main_cities_lists.json must be a JSON object mapping country -> list of cities.")
            return {}
        out: Dict[str, List[str]] = {}
        for k, v in raw.items():
            if not isinstance(k, str) or not isinstance(v, list):
                continue
            cleaned = []
            seen = set()
            for c in v:
                if not isinstance(c, str):
                    continue
                cc = re.sub(r"\s+", " ", c).strip()
                if cc and cc.casefold() not in seen:
                    seen.add(cc.casefold())
                    cleaned.append(cc)
            out[k] = sorted(cleaned)
        return out
    except Exception as e:
        st.error(f"Failed to parse main_cities_lists.json: {e}")
        return {}

# ------------------------------
# Sidebar Navigation
# ------------------------------

st.sidebar.title("🌍 Job Location Analyzer")
st.sidebar.caption("Analyze multi-sheet Excel of job vacancy locations")

with st.sidebar:
    st.markdown("**1) Upload the Excel file**")
    uploaded_file = st.file_uploader("Excel (.xlsx)", type=["xlsx"], help="Each sheet = one country + 'Unknown_Country'")
    st.markdown("**2) (Optional) Upload `main_cities_lists.json`**")
    cities_json_file = st.file_uploader(
        "main_cities_lists.json",
        type=["json"],
        help="JSON mapping: { 'Lebanon': ['Beirut','Tripoli','Sour', ...], ... }"
    )
    st.divider()
    page = st.radio(
        "Pages",
        options=[
            "📥 Load & Overview",
            "🔍 Sheet Explorer",
            "Sheet Explorer V2",      # NEW PAGE
            "⚖️ Compare Countries",
            "🌍 Geo Map",
            "🧹 Data Quality",
            "📊 Pivot & Export",
            "❓ Help",
        ],
        index=0,
    )
    st.divider()
    st.markdown("Built for ESCWA pipelines • v1.1")

if not uploaded_file:
    st.info("⬅️ Upload an Excel file to begin.")
    st.stop()

# ------------------------------
# Load data
# ------------------------------
dfs = read_excel_all_sheets(uploaded_file)
if not dfs:
    st.stop()

# Load (optional) main cities JSON
main_cities_map: Dict[str, List[str]] = {}
if cities_json_file is not None:
    main_cities_map = _load_main_cities_json_from_bytes(cities_json_file.getvalue())

all_df = get_all_data(dfs)
overview = aggregated_overview(dfs)
centroids = country_centroids(dfs)

unknown_sheet_name = "Unknown_Country" if "Unknown_Country" in dfs else None

# ------------------------------
# Pages
# ------------------------------

if page == "📥 Load & Overview":
    st.header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Sheets (countries)", len(dfs))
    with c2:
        st.metric("Total rows", f"{len(all_df):,}")
    with c3:
        total_posts = int(all_df["count"].sum()) if not all_df.empty else 0
        st.metric("Total job posts", f"{total_posts:,}")
    with c4:
        unknown_rows = len(dfs.get("Unknown_Country", pd.DataFrame()))
        st.metric("Unknown rows", f"{unknown_rows:,}")

    st.subheader("Per-country summary")
    st.dataframe(overview, use_container_width=True, hide_index=True)

    st.subheader("Language mix (Arabic vs Latin) by Sheet")

    if all_df.empty:
        lang = pd.DataFrame(columns=["Sheet", "Arabic in City (share)", "Arabic in SRC_Country (share)"])
    else:
        lang = (
            all_df.groupby("Sheet", as_index=False)
            .agg(
                **{
                    "Arabic in City (share)": ("is_arabic_city", lambda s: float(s.mean()) * 100.0),
                    "Arabic in SRC_Country (share)": ("is_arabic_country", lambda s: float(s.mean()) * 100.0),
                }
            )
            .round({"Arabic in City (share)": 2, "Arabic in SRC_Country (share)": 2})
        )

    st.dataframe(lang.sort_values("Arabic in City (share)", ascending=False), use_container_width=True, hide_index=True)

elif page == "🔍 Sheet Explorer":
    st.header("Sheet Explorer")
    sheet = st.selectbox("Select a sheet (country)", options=list(dfs.keys()))
    df = dfs[sheet]
    st.caption(f"Rows: {len(df):,} • Total posts: {int(df['count'].sum()):,} • Unique cities: {df['City'].nunique():,}")

    # Aggregations
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Cities by Job Posts")
        topn = st.slider("Top N", min_value=5, max_value=50, value=15, key="topn_city")
        city_agg = (
            df.groupby(["City", "Latitude", "Longitude"], as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .head(topn)
        )
        st.dataframe(city_agg, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("SRC_Country Variants (as seen in raw posts)")
        topn2 = st.slider("Top N variants", min_value=5, max_value=50, value=15, key="topn_src")
        country_agg = (
            df.groupby("SRC_Country", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
            .head(topn2)
        )
        st.dataframe(country_agg, use_container_width=True, hide_index=True)

    st.subheader("Geo scatter (weighted by count)")

    # Build a separate aggregation for the MAP (use ALL rows, not just Top-N)
    map_agg = (
        df.groupby(["City", "Latitude", "Longitude"], as_index=False)["count"]
        .sum()
        .rename(columns={"Latitude": "lat", "Longitude": "lon"})
    )

    if map_agg.empty:
        st.info("No geocoded rows to plot for this sheet.")
    else:
        # Controls for size scaling (keys made per-sheet so they don't clash)
        cmin, cmax = st.columns(2)
        with cmin:
            sheet_scale_mode = st.radio(
                "Size scale",
                ["sqrt", "linear", "log"],
                index=0,
                key=f"{sheet}_scale_mode",
                horizontal=True,
            )
        with cmax:
            sheet_min_r = st.slider("Min radius (m)", 100, 10000, 1000, key=f"{sheet}_min_r")
            sheet_max_r = st.slider("Max radius (m)", 1000, 30000, 8000, key=f"{sheet}_max_r")

        map_agg["radius"] = _scale_radius(map_agg["count"], sheet_min_r, sheet_max_r, sheet_scale_mode)
        map_agg["color"] = [color_for_sheet(sheet)] * len(map_agg)

        # Safe view state (mean center)
        view_state = pdk.ViewState(
            latitude=float(map_agg["lat"].mean()),
            longitude=float(map_agg["lon"].mean()),
            zoom=4,
        )

        deck = pdk.Deck(
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_agg,
                    get_position="[lon, lat]",
                    get_radius="radius",
                    get_fill_color="color",
                    pickable=True,
                    opacity=0.6,
                    auto_highlight=True,
                )
            ],
            initial_view_state=view_state,
            tooltip={"text": f"{sheet} • {{City}}\n{{count}} posts"},
            height=480,  # keep a stable footprint
        )
        st.pydeck_chart(deck, use_container_width=True, key=f"{sheet}_overview_map")

    # === City drill-down (place BEFORE "Potential anomalies") ===
    st.subheader("City drill-down")

    # Build city list (exclude blanks/NaNs)
    city_options = sorted(
        [c for c in df["City"].dropna().astype(str).str.strip().unique() if c != ""]
    )
    if not city_options:
        st.info("No cities available in this sheet.")
    else:
        sel_city = st.selectbox("Select a city", options=city_options, key="sel_city")

        city_df = df[df["City"] == sel_city].copy()

        total_city_posts = int(city_df["count"].sum())
        n_variants = int(city_df["SRC_Country"].nunique())
        n_coords = int(city_df[["Latitude", "Longitude"]].drop_duplicates().shape[0])

        m1, m2, m3 = st.columns(3)
        m1.metric("Total posts (city)", f"{total_city_posts:,}")
        m2.metric("SRC_Country variants", f"{n_variants:,}")
        m3.metric("Unique coordinates", f"{n_coords:,}")

        st.markdown("**SRC_Country breakdown for the selected city**")
        src_agg = (
            city_df.groupby("SRC_Country", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
        )
        src_agg["% of city"] = (src_agg["count"] / max(total_city_posts, 1) * 100).round(2)
        st.dataframe(src_agg, use_container_width=True, hide_index=True)

        st.bar_chart(src_agg, x="SRC_Country", y="count", use_container_width=True)

        # --- Nearby posts within a radius of the selected city ---
        st.subheader("Nearby posts around this city")

        coords_w = (
            city_df.groupby(["Latitude", "Longitude"], as_index=False)["count"].sum()
        )
        if coords_w.empty:
            st.info("No coordinates available for this city to compute nearby posts.")
        else:
            if coords_w["count"].sum() > 0:
                lat0 = float(np.average(coords_w["Latitude"], weights=coords_w["count"]))
                lon0 = float(np.average(coords_w["Longitude"], weights=coords_w["count"]))
            else:
                lat0 = float(coords_w["Latitude"].mean())
                lon0 = float(coords_w["Longitude"].mean())

            radius_km = st.slider(
                "Radius around city (km)",
                min_value=5, max_value=500, value=50, step=5,
                help="Show all posts in this sheet within the selected distance from the city center.",
                key=f"{sheet}_{sel_city}_radius",
            )

            base_df = df.copy()
            dist_arr = _haversine_km_vec(
                base_df["Latitude"].to_numpy(),
                base_df["Longitude"].to_numpy(),
                lat0, lon0
            )
            base_df["distance_km"] = dist_arr

            near_df = base_df[base_df["distance_km"] <= float(radius_km)].copy()

            if near_df.empty:
                st.info("No posts found within the selected radius.")
            else:
                near_geo = (
                    near_df.groupby(["City", "Latitude", "Longitude"], as_index=False)
                    .agg(count=("count", "sum"), distance_km=("distance_km", "min"))
                    .rename(columns={"Latitude": "lat", "Longitude": "lon"})
                    .sort_values(["distance_km", "count"], ascending=[True, False])
                )

                csz1, csz2, csz3 = st.columns(3)
                with csz1:
                    near_scale_mode = st.radio(
                        "Size scale",
                        ["sqrt", "linear", "log"],
                        index=0,
                        key=f"{sheet}_{sel_city}_near_scale",
                        horizontal=True,
                    )
                with csz2:
                    near_min_r = st.slider(
                        "Min radius (m)",
                        min_value=100, max_value=10000, value=600, step=50,
                        key=f"{sheet}_{sel_city}_near_minr",
                    )
                with csz3:
                    near_max_r = st.slider(
                        "Max radius (m)",
                        min_value=1000, max_value=30000, value=6000, step=100,
                        key=f"{sheet}_{sel_city}_near_maxr",
                    )
                if near_max_r <= near_min_r:
                    near_max_r = near_min_r + 1

                near_geo["radius"] = _scale_radius(near_geo["count"], near_min_r, near_max_r, near_scale_mode)
                near_geo["color"] = [color_for_sheet(sheet)] * len(near_geo)

                if radius_km <= 10:
                    zoom_lvl = 10
                elif radius_km <= 25:
                    zoom_lvl = 9
                elif radius_km <= 50:
                    zoom_lvl = 8
                elif radius_km <= 100:
                    zoom_lvl = 7
                elif radius_km <= 200:
                    zoom_lvl = 6
                elif radius_km <= 350:
                    zoom_lvl = 5
                else:
                    zoom_lvl = 4

                nearby_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=near_geo,
                    get_position="[lon, lat]",
                    get_radius="radius",
                    get_fill_color="color",
                    pickable=True,
                    opacity=0.6,
                    auto_highlight=True,
                )
                nearby_view = pdk.ViewState(
                    latitude=lat0,
                    longitude=lon0,
                    zoom=zoom_lvl,
                )

                st.pydeck_chart(
                    pdk.Deck(
                        layers=[nearby_layer],
                        initial_view_state=nearby_view,
                        tooltip={"text": f"{sheet} • {{City}}\n{{count}} posts\n{{distance_km}} km from center"},
                        height=460,
                    ),
                    use_container_width=True,
                    key=f"{sheet}_{sel_city}_nearby_map",
                )

                st.markdown("**Posts within radius (by city & coordinate)**")
                st.dataframe(
                    near_geo[["City", "lat", "lon", "distance_km", "count"]]
                        .rename(columns={"lat": "Latitude", "lon": "Longitude"})
                        .reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

# ------------------------------
# NEW PAGE: Sheet Explorer V2
# ------------------------------
elif page == "Sheet Explorer V2":
    st.header("Sheet Explorer V2")

    if not main_cities_map:
        st.info("Upload **main_cities_lists.json** in the sidebar to enable city selection for this page.")
    # Offer country options = intersection between sheets and JSON keys (if JSON provided), otherwise all sheets
    if main_cities_map:
        available_sheets = sorted(set(dfs.keys()).intersection(main_cities_map.keys()))
        if not available_sheets:
            st.warning("No overlap between Excel sheet names and keys in main_cities_lists.json. Showing all sheets.")
            available_sheets = sorted(dfs.keys())
    else:
        available_sheets = sorted(dfs.keys())

    sheet2 = st.selectbox("Country (sheet)", options=available_sheets, key="v2_sheet")
    df2 = dfs[sheet2]

    # Build city choices from the JSON for the selected country
    json_cities: List[str] = main_cities_map.get(sheet2, [])
    city_choices = ["All"] + json_cities if json_cities else ["All"]
    sel_city2 = st.selectbox("City (from main_cities_lists.json)", options=city_choices, key="v2_city")

    # Filter the dataframe based on chosen city (case-insensitive)
    df2 = df2.copy()
    df2["__City_norm__"] = df2["City"].apply(_normcase)

    if sel_city2 != "All":
        target_norm = _normcase(sel_city2)
        filtered = df2[df2["__City_norm__"] == target_norm]
    else:
        filtered = df2

    st.caption(
        f"Rows: {len(filtered):,} • Total posts: {int(filtered['count'].sum()):,} • "
        f"Unique cities: {filtered['City'].nunique():,}"
    )

    # Aggregate for the map
    map2 = (
        filtered.groupby(["City", "Latitude", "Longitude"], as_index=False)["count"]
        .sum()
        .rename(columns={"Latitude": "lat", "Longitude": "lon"})
        .sort_values("count", ascending=False)
    )

    if sel_city2 != "All" and map2.empty:
        st.warning("No rows matched this city in the selected sheet. Check spellings/variants in your data.")
    elif map2.empty:
        st.info("No geocoded rows to plot for this selection.")
    else:
        # Sizing controls
        csa, csb, csc = st.columns(3)
        with csa:
            v2_scale_mode = st.radio(
                "Size scale",
                ["sqrt", "linear", "log"],
                index=0,
                key=f"{sheet2}_v2_scale",
                horizontal=True,
            )
        with csb:
            v2_min_r = st.slider("Min radius (m)", 100, 10000, 800, key=f"{sheet2}_v2_minr")
        with csc:
            v2_max_r = st.slider("Max radius (m)", 1000, 30000, 7000, key=f"{sheet2}_v2_maxr")
        if v2_max_r <= v2_min_r:
            v2_max_r = v2_min_r + 1

        map2["radius"] = _scale_radius(map2["count"], v2_min_r, v2_max_r, v2_scale_mode)
        map2["color"] = [color_for_sheet(sheet2)] * len(map2)

        # View state: zoom tighter if a single city is chosen
        if sel_city2 != "All":
            # Weighted center on this city selection
            if map2["count"].sum() > 0:
                vlat = float(np.average(map2["lat"], weights=map2["count"]))
                vlon = float(np.average(map2["lon"], weights=map2["count"]))
            else:
                vlat = float(map2["lat"].mean())
                vlon = float(map2["lon"].mean())
            zoom2 = 8
        else:
            vlat = float(map2["lat"].mean())
            vlon = float(map2["lon"].mean())
            zoom2 = 4

        layer2 = pdk.Layer(
            "ScatterplotLayer",
            data=map2,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            opacity=0.6,
            auto_highlight=True,
        )
        view2 = pdk.ViewState(latitude=vlat, longitude=vlon, zoom=zoom2)

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer2],
                initial_view_state=view2,
                tooltip={"text": f"{sheet2} • {{City}}\n{{count}} posts"},
                height=480,
            ),
            use_container_width=True,
            key=f"{sheet2}_v2_map",
        )

        st.subheader("Table (points shown on map)")
        st.dataframe(
            map2.rename(columns={"lat": "Latitude", "lon": "Longitude"}),
            use_container_width=True,
            hide_index=True
        )

elif page == "⚖️ Compare Countries":
    st.header("Compare Countries")
    choices = list(dfs.keys())
    compare = st.multiselect("Select 2–6 sheets to compare", options=choices, default=[c for c in choices[:2]])
    if len(compare) < 2:
        st.info("Select at least two sheets.")
        st.stop()

    subset = all_df[all_df["Sheet"].isin(compare)]
    st.subheader("Totals")
    totals = subset.groupby("Sheet", as_index=False)["count"].sum().sort_values("count", ascending=False)
    st.bar_chart(totals, x="Sheet", y="count", use_container_width=True)

    st.subheader("Top cities in selected sheets")
    topn = st.slider("Top N per sheet", 5, 50, 10, key="cmp_topn")
    top_cities = (
        subset.groupby(["Sheet", "City"], as_index=False)["count"]
        .sum()
        .sort_values(["Sheet", "count"], ascending=[True, False])
        .groupby("Sheet", group_keys=False)
        .head(topn)
    )
    st.dataframe(top_cities, use_container_width=True, hide_index=True)

    st.subheader("City set overlap (Jaccard)")
    city_sets = {s: set(dfs[s]["City"].unique()) for s in compare}
    rows = []
    for i, s1 in enumerate(compare):
        for s2 in compare[i+1:]:
            rows.append({"Sheet A": s1, "Sheet B": s2, "Jaccard(city names)": round(jaccard_similarity(city_sets[s1], city_sets[s2]), 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif page == "🌍 Geo Map":
    st.header("Global Map")
    sel = st.multiselect("Sheets to display", options=list(dfs.keys()), default=list(dfs.keys()))
    data = all_df[all_df["Sheet"].isin(sel)].copy()
    if data.empty:
        st.info("No data.")
        st.stop()

    city_geo = (
        data.groupby(["Sheet", "City", "Latitude", "Longitude"], as_index=False)["count"]
        .sum()
        .rename(columns={"Latitude":"lat","Longitude":"lon"})
    )

    st.caption("Circle size reflects post counts")
    g1, g2 = st.columns(2)
    with g1:
        globe_scale_mode = st.radio("Size scale", ["sqrt", "linear", "log"], index=0, key="globe_scale_mode", horizontal=True)
    with g2:
        globe_min_r = st.slider("Min radius (m)", 100, 10000, 1000, key="globe_min_r")
        globe_max_r = st.slider("Max radius (m)", 1000, 40000, 10000, key="globe_max_r")

    city_geo["radius"] = _scale_radius(city_geo["count"], globe_min_r, globe_max_r, globe_scale_mode)
    city_geo["color"] = city_geo["Sheet"].apply(color_for_sheet)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=city_geo,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        opacity=0.6,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(latitude=float(city_geo["lat"].mean()), longitude=float(city_geo["lon"].mean()), zoom=3)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Sheet} • {City}\n{count} posts"})
    st.pydeck_chart(r)

    st.subheader("Table")
    st.dataframe(city_geo.sort_values("count", ascending=False), use_container_width=True, hide_index=True)

elif page == "🧹 Data Quality":
    st.header("Data Quality & Normalization Preview")

    st.subheader("Anomalies — coordinates likely belong to another sheet")

    c1, c2, c3 = st.columns(3)
    with c1:
        own_threshold_km = st.slider(
            "Own-sheet distance threshold (km)",
            min_value=50, max_value=2000, value=200, step=10,
            help="Only flag if a row is farther than this from its own sheet's centroid."
        )
    with c2:
        margin_km = st.slider(
            "Margin advantage for other sheet (km)",
            min_value=20, max_value=1000, value=100, step=10,
            help="Nearest other centroid must be at least this much closer than the own centroid."
        )
    with c3:
        exclude_unknown = st.checkbox(
            "Exclude Unknown_Country from checks",
            value=True,
            help="If checked, don't evaluate Unknown_Country and don't consider it as a target centroid."
        )

    # Build centroid lookup table just-in-time
    def _compute_anomalies(
        dfs: Dict[str, pd.DataFrame],
        centroids_df: pd.DataFrame,
        own_threshold_km: float,
        margin_km: float,
        exclude_unknown: bool,
    ) -> pd.DataFrame:
        if centroids_df.empty:
            return pd.DataFrame()

        centroids_df = centroids_df.copy().rename(columns={"centroid_lat": "clat", "centroid_lon": "clon"})
        centroid_map = {r["Sheet"]: (r["clat"], r["clon"]) for _, r in centroids_df.iterrows()}

        sheets = list(dfs.keys())
        if exclude_unknown:
            sheets = [s for s in sheets if s != "Unknown_Country"]
            centroids_candidates = centroids_df[centroids_df["Sheet"] != "Unknown_Country"].reset_index(drop=True)
        else:
            centroids_candidates = centroids_df.reset_index(drop=True)

        cand_names = centroids_candidates["Sheet"].to_list()
        cand_lats = centroids_candidates["clat"].to_numpy()
        cand_lons = centroids_candidates["clon"].to_numpy()

        out_frames: List[pd.DataFrame] = []
        for sheet in sheets:
            df = dfs.get(sheet)
            if df is None or df.empty:
                continue
            if sheet not in centroid_map:
                continue

            self_lat, self_lon = centroid_map[sheet]
            lat_arr = df["Latitude"].to_numpy()
            lon_arr = df["Longitude"].to_numpy()

            self_d = _haversine_km_vec(lat_arr, lon_arr, self_lat, self_lon)
            dists = np.vstack([
                _haversine_km_vec(lat_arr, lon_arr, cand_lats[j], cand_lons[j])
                for j in range(len(cand_names))
            ]).T

            nearest_idx = dists.argmin(axis=1)
            nearest_dist = dists[np.arange(len(lat_arr)), nearest_idx]
            nearest_sheet = np.array(cand_names, dtype=object)[nearest_idx]

            mask = (
                (nearest_sheet != sheet) &
                (self_d > own_threshold_km) &
                (nearest_dist + margin_km < self_d)
            )

            if mask.any():
                sub = df.loc[mask, ["SRC_Country", "City", "Latitude", "Longitude", "count"]].copy()
                sub.insert(0, "Sheet", sheet)
                sub["self_dist_km"] = np.round(self_d[mask], 2)
                sub["nearest_sheet"] = nearest_sheet[mask]
                sub["nearest_dist_km"] = np.round(nearest_dist[mask], 2)
                sub["distance_advantage_km"] = np.round(sub["self_dist_km"] - sub["nearest_dist_km"], 2)
                out_frames.append(sub)

        if not out_frames:
            return pd.DataFrame(columns=[
                "Sheet", "SRC_Country", "City", "Latitude", "Longitude", "count",
                "self_dist_km", "nearest_sheet", "nearest_dist_km", "distance_advantage_km"
            ])

        out = pd.concat(out_frames, ignore_index=True)
        out = out.sort_values(["distance_advantage_km", "self_dist_km"], ascending=[False, False], ignore_index=True)
        return out

    anomalies = _compute_anomalies(
        dfs=dfs,
        centroids_df=centroids,
        own_threshold_km=own_threshold_km,
        margin_km=margin_km,
        exclude_unknown=exclude_unknown,
    )

    if anomalies.empty:
        st.success("No cross-sheet coordinate anomalies found under the current thresholds.")
    else:
        st.caption(
            "Flagged rows are significantly closer to another sheet's centroid than their own. "
            "Use sliders above to refine sensitivity."
        )

        summary = anomalies.groupby(["Sheet", "nearest_sheet"], as_index=False).size().rename(columns={"size": "rows"})
        st.dataframe(summary.sort_values("rows", ascending=False), use_container_width=True, hide_index=True)

        st.dataframe(
            anomalies[
                ["Sheet", "nearest_sheet", "SRC_Country", "City", "Latitude", "Longitude",
                 "count", "self_dist_km", "nearest_dist_km", "distance_advantage_km"]
            ],
            use_container_width=True, hide_index=True
        )

        st.download_button(
            "⬇️ Download anomalies CSV",
            data=anomalies.to_csv(index=False).encode("utf-8"),
            file_name="cross_sheet_coordinate_anomalies.csv",
            mime="text/csv",
        )

elif page == "📊 Pivot & Export":
    st.header("Pivot & Export")
    st.info("This placeholder page can be filled with your pivot/export tools as needed.")

elif page == "❓ Help":
    st.header("How to Use")
    st.markdown(
        """
        **Input format**: one Excel file where _each sheet_ corresponds to a country, plus optionally *Unknown_Country*.
        Every sheet should contain columns:
        - `SRC_Country` - raw country/place string as scraped from the job post
        - `City` - city/area/place name
        - `Latitude`, `Longitude` - coordinates of the city (float)
        - `count` - number of job posts (can contain commas; handled automatically)

        **New in v1.1**
        - **Sheet Explorer V2** uses `main_cities_lists.json` to drive the city dropdown. Choose a country, then “All” or a city
          from your curated list to focus the map.

        **Tips**
        - If your column names are slightly different (e.g., `lat` or `SRC country`), the app will auto-align common variants.
        - Keep sheet names as country names (plus *Unknown_Country* for entries you cannot classify).
        - You can filter/sort any table and download prepared CSVs.
        """
    )
