import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import ahocorasick
from json import load
from xlsxwriter import Workbook
from functools import lru_cache

# ==========================
# Constants
# ==========================
UNKNOWN_COUNTRY = "Unknown_Country"

# ==========================
# Precompiled regexes & tables
# ==========================
TOKEN_SPLIT_RE = re.compile(r"[\s\-\u2010-\u2015\u2212_]+")
INVALID_SHEET_CHARS_RE = re.compile(r'[:\\/\?\*\[\]]')
COMBINING_RE = re.compile(
    r"[\u0300-\u036F\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+"
)
TRANS_TABLE = str.maketrans({
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0622": "\u0627",  # آ -> ا
    "\u0649": "\u064A",  # ى -> ي
    "\u0629": "\u0647",  # ة -> ه
    "\u0640": "",        # tatweel
    "\u200c": "",        # ZWNJ
    "\u200d": "",        # ZWJ
    "\ufeff": "",        # BOM
})

CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[A-Z])(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])"
)

# Module-level cache
_CAMEL_CACHE: Dict[str, Optional[str]] = {}

# ==========================
# Unicode/Arabic helpers
# ==========================
def _is_arabic_char(ch: str) -> bool:
    return (
        "\u0600" <= ch <= "\u06FF"
        or "\u0750" <= ch <= "\u077F"
        or "\u08A0" <= ch <= "\u08FF"
        or "\uFB50" <= ch <= "\uFDFF"
        or "\uFE70" <= ch <= "\uFEFF"
    )

def _is_arabic_token(tok: str) -> bool:
    return any(_is_arabic_char(c) for c in tok)

def _strip_al(tok: str) -> str:
    return tok[2:] if tok.startswith("ال") else tok

def _add_al(tok: str) -> str:
    if _is_arabic_token(tok) and not tok.startswith("ال"):
        return "ال" + tok
    return tok

@lru_cache(maxsize=20_000)
def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = COMBINING_RE.sub("", s)
    s = s.translate(TRANS_TABLE)
    return s.casefold()

@lru_cache(maxsize=10_000)
def _variants_for_name(name: str) -> frozenset[str]:
    """
    Build Unicode-aware variants with full coverage for better recall.
    """
    base = _normalize_text(name.strip())
    if not base:
        return frozenset()

    tokens = [t for t in TOKEN_SPLIT_RE.split(base) if t]
    if not tokens:
        return frozenset()

    tokens_no_al = [_strip_al(t) for t in tokens]
    tokens_with_al = [_add_al(t) for t in tokens]

    variants_tokens: List[List[str]] = [
        tokens,
        tokens_no_al,
        tokens_with_al,
    ]

    # Add first-token variations for Arabic
    if len(tokens) > 1 and any(_is_arabic_token(t) for t in tokens):
        t_first_strip = [_strip_al(tokens[0])] + tokens[1:]
        t_first_add = [_add_al(tokens[0])] + tokens[1:]
        variants_tokens.extend([t_first_strip, t_first_add])

    raw_forms: set[str] = set()
    for toks in variants_tokens:
        toks = [t for t in toks if t]
        if not toks:
            continue
        raw_forms.add(" ".join(toks))
        raw_forms.add("-".join(toks))
        raw_forms.add("".join(toks))

    return frozenset(raw_forms)

def _is_word_char(ch: str) -> bool:
    return ch.isalnum()

def _is_boundary(s: str, i: int) -> bool:
    return i < 0 or i >= len(s) or not _is_word_char(s[i])

def _maybe_camel_split(original: str) -> Optional[str]:
    """Return a space-split version at camelCase boundaries if any were found; else None."""
    if original in _CAMEL_CACHE:
        return _CAMEL_CACHE[original]
    
    if not isinstance(original, str) or not original:
        _CAMEL_CACHE[original] = None
        return None
    
    if CAMEL_BOUNDARY_RE.search(original) is None:
        _CAMEL_CACHE[original] = None
        return None
    
    result = CAMEL_BOUNDARY_RE.sub(" ", original)
    
    # Limit cache size
    if len(_CAMEL_CACHE) > 5000:
        _CAMEL_CACHE.clear()
    
    _CAMEL_CACHE[original] = result
    return result

def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

# ==========================
# Automaton builders
# ==========================
def build_country_automaton(country_names: Dict[str, List[str]]) -> ahocorasick.Automaton:
    """Build country automaton."""
    AC = ahocorasick.Automaton()
    added: set[str] = set()

    for country_key, aliases in country_names.items():
        if not isinstance(country_key, str):
            continue
        forms = [country_key]
        if isinstance(aliases, list):
            forms += [a for a in aliases if isinstance(a, str)]

        for raw in forms:
            for var in _variants_for_name(raw):
                if var in added:
                    continue
                added.add(var)
                AC.add_word(var, (country_key, len(var)))

    AC.make_automaton()
    return AC

def build_places_automatons_v2(cities_dict: Dict[str, List[Dict[str, Any]]]
                               ) -> Tuple[Dict[str, ahocorasick.Automaton], ahocorasick.Automaton]:
    per_country: Dict[str, ahocorasick.Automaton] = {}
    global_ac = ahocorasick.Automaton()
    global_added: set[str] = set()

    for country, places in cities_dict.items():
        if not isinstance(country, str) or not isinstance(places, list):
            continue
        norm_country = _normalize_text(country)

        local_ac = ahocorasick.Automaton()
        local_added: set[str] = set()

        for place in places:
            if not isinstance(place, dict):
                continue

            original_name = str(place.get("name", "")).strip()
            if not original_name:
                continue

            plat = _to_float(place.get("latitude"))
            plon = _to_float(place.get("longitude"))
            pvars = place.get("name_variants", []) or []

            all_search_terms = [original_name] + [str(v) for v in pvars if isinstance(v, str)]
            # keep raw-name equivalence for completeness
            base_is_country_equiv = _normalize_text(original_name) == norm_country

            seen: set[str] = set()
            for search_term in all_search_terms:
                for var in _variants_for_name(search_term):
                    if var in seen:
                        continue
                    seen.add(var)

                    # variant-level equivalence (defense-in-depth)
                    is_equiv = base_is_country_equiv or (var == norm_country)

                    if var not in local_added:
                        local_ac.add_word(var, (original_name, plat, plon, len(var), is_equiv))
                        local_added.add(var)

                    if var not in global_added:
                        global_ac.add_word(var, (country, original_name, plat, plon, len(var)))
                        global_added.add(var)

        local_ac.make_automaton()
        per_country[country] = local_ac

    global_ac.make_automaton()
    return per_country, global_ac

def build_cities_coor_automatons(cities_coor: Dict[str, List[Dict[str, Any]]]
                                 ) -> Tuple[Dict[str, ahocorasick.Automaton], ahocorasick.Automaton]:
    per_country: Dict[str, ahocorasick.Automaton] = {}
    global_ac = ahocorasick.Automaton()
    global_added: set[str] = set()

    for country, places in cities_coor.items():
        if not isinstance(country, str) or not isinstance(places, list):
            continue
        norm_country = _normalize_text(country)

        local_ac = ahocorasick.Automaton()
        local_added: set[str] = set()

        for place in places:
            if not isinstance(place, dict):
                continue

            raw_name = str(place.get("name", "")).strip()
            if not raw_name:
                continue

            canonical = place.get("canonical")
            canonical = str(canonical).strip() if isinstance(canonical, str) else ""
            canonical_or_name = canonical if canonical else raw_name

            plat = _to_float(place.get("latitude"))
            plon = _to_float(place.get("longitude"))

            base_is_country_equiv = _normalize_text(raw_name) == norm_country

            for var in _variants_for_name(raw_name):
                is_equiv = base_is_country_equiv or (var == norm_country)

                if var not in local_added:
                    local_ac.add_word(var, (canonical_or_name, plat, plon, len(var), is_equiv))
                    local_added.add(var)

                if var not in global_added:
                    global_ac.add_word(var, (country, canonical_or_name, plat, plon, len(var)))
                    global_added.add(var)

        local_ac.make_automaton()
        per_country[country] = local_ac

    global_ac.make_automaton()
    return per_country, global_ac

def _same_name(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    return _normalize_text(a) == _normalize_text(b)

# ==========================
# CountryCityDetector
# ==========================
class CountryCityDetector:
    def __init__(self,
                 cities_file_v2: str = "data/main_json.json",
                 country_names_file: str = "data/country_names.json",
                 cities_coor_file: str = "data/main_cities_coor.json"):
        
        print("Loading JSON files...")
        self.cities_v2 = self._load_json(cities_file_v2)
        self.country_names = self._load_json(country_names_file)
        self.cities_coor = self._load_json(cities_coor_file)

        print("Building automatons...")
        self.AC_COUNTRY = build_country_automaton(self.country_names)
        self.AC_COOR_PER, self.AC_COOR_GLOBAL = build_cities_coor_automatons(self.cities_coor)
        self.AC_V2_PER, self.AC_V2_GLOBAL = build_places_automatons_v2(self.cities_v2)

        # Default coords per country
        self.country_default_coords: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for c, places in self.cities_v2.items():
            norm_c = _normalize_text(c)
            latlon = (None, None)
            if isinstance(places, list):
                for p in places:
                    if not isinstance(p, dict):
                        continue
                    n = str(p.get("name", "")).strip()
                    if _normalize_text(n) == norm_c:
                        latlon = (_to_float(p.get("latitude")), _to_float(p.get("longitude")))
                        break
            self.country_default_coords[c] = latlon

    @staticmethod
    def _load_json(file: str) -> Dict[str, Any]:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = load(f)
                return data if isinstance(data, dict) else {}
        except Exception as err:
            print(f"Failed to load JSON file {file}: {err}")
            return {}

    @staticmethod
    def _find_country(text_norm: str, ac: ahocorasick.Automaton) -> Optional[str]:
        """Search for LONGEST boundary-valid country match."""
        best_country = None
        best_len = -1
        for end_pos, payload in ac.iter(text_norm):
            country, klen = payload
            start = end_pos - klen + 1
            if _is_boundary(text_norm, start - 1) and _is_boundary(text_norm, end_pos + 1):
                if klen > best_len:
                    best_len = klen
                    best_country = country
        return best_country

    @staticmethod
    def _find_place_in_country(ac: Optional[ahocorasick.Automaton], text_norm: str
                               ) -> Optional[Tuple[str, Optional[float], Optional[float]]]:
        """Generic per-country search."""
        if not ac:
            return None
        best = None
        best_len = -1
        for end_pos, payload in ac.iter(text_norm):
            display_name, plat, plon, klen, is_country_equiv = payload
            if is_country_equiv:
                continue
            start = end_pos - klen + 1
            if _is_boundary(text_norm, start - 1) and _is_boundary(text_norm, end_pos + 1):
                if klen > best_len:
                    best_len = klen
                    best = (display_name, plat, plon)
        return best

    @staticmethod
    def _find_any_place(ac: ahocorasick.Automaton, text_norm: str
                        ) -> Optional[Tuple[str, str, Optional[float], Optional[float]]]:
        """Generic global search."""
        best = None
        best_len = -1
        for end_pos, payload in ac.iter(text_norm):
            country, display_name, plat, plon, klen = payload
            start = end_pos - klen + 1
            if _is_boundary(text_norm, start - 1) and _is_boundary(text_norm, end_pos + 1):
                if klen > best_len:
                    best_len = klen
                    best = (country, display_name, plat, plon)
        return best

    def detect(self, text: Any) -> Dict[str, Any]:
        """
        Main detection logic following the spec exactly:
        1. Try country-first with phrases, then full text scan
        2. If no country, try global city search
        """
        original = text if isinstance(text, str) else ""
        candidates: List[str] = [_normalize_text(original)]

        # Try CamelCase split as additional candidate
        split = _maybe_camel_split(original)
        if split:
            cand2 = _normalize_text(split)
            if cand2 != candidates[0]:
                candidates.append(cand2)

        # Early exclusion rules
        for cand in candidates:
            if _has_word(cand, "Sur") and _has_word(cand, "France"):
                return {
                    "Country": UNKNOWN_COUNTRY,
                    "City": None,
                    "Latitude": None,
                    "Longitude": None,
                }
            if _has_word(cand, "Lebanon") and _has_word(cand, "Pennsylvania"):
                return {
                    "Country": UNKNOWN_COUNTRY,
                    "City": None,
                    "Latitude": None,
                    "Longitude": None,
                }

        # Stage 1: Country-first approach
        for cand in candidates:
            country = self._find_country(cand, self.AC_COUNTRY)
            if country:
                # Build candidate phrases from ORIGINAL text (before full normalization)
                phrases = _candidate_phrases(original)
                
                # Try exact phrase matching first (preferred source)
                for ph in phrases:
                    place = _exact_phrase_in_country(self.AC_COOR_PER.get(country), ph)
                    if place:
                        city_name, plat, plon = place
                        if _same_name(city_name, country):
                            dlat, dlon = self.country_default_coords.get(country, (None, None))
                            return {"Country": country, "City": None, "Latitude": dlat, "Longitude": dlon}
                        return {"Country": country, "City": city_name, "Latitude": plat, "Longitude": plon}
                                    
                # Try exact phrase matching (fallback source with name_variants)
                for ph in phrases:
                    place = _exact_phrase_in_country(self.AC_V2_PER.get(country), ph)
                    if place:
                        city_name, plat, plon = place
                        if _same_name(city_name, country):
                            dlat, dlon = self.country_default_coords.get(country, (None, None))
                            return {"Country": country, "City": None, "Latitude": dlat, "Longitude": dlon}
                        return {"Country": country, "City": city_name, "Latitude": plat, "Longitude": plon}

                # Full text scan (preferred source)
                place = self._find_place_in_country(self.AC_COOR_PER.get(country), cand)
                if place:
                    city_name, plat, plon = place
                    if _same_name(city_name, country):
                        dlat, dlon = self.country_default_coords.get(country, (None, None))
                        return {"Country": country, "City": None, "Latitude": dlat, "Longitude": dlon}
                    return {"Country": country, "City": city_name, "Latitude": plat, "Longitude": plon}

                # Full text scan (fallback source with name_variants)
                place = self._find_place_in_country(self.AC_V2_PER.get(country), cand)
                if place:
                    city_name, plat, plon = place
                    if _same_name(city_name, country):
                        dlat, dlon = self.country_default_coords.get(country, (None, None))
                        return {"Country": country, "City": None, "Latitude": dlat, "Longitude": dlon}
                    return {"Country": country, "City": city_name, "Latitude": plat, "Longitude": plon}

                # Country found but no city - return with default coords
                plat, plon = self.country_default_coords.get(country, (None, None))
                return {
                    "Country": country,
                    "City": None,
                    "Latitude": plat,
                    "Longitude": plon
                }

        # Stage 2: No country found - global city search
        for cand in candidates:
            # Try preferred source
            any_place = self._find_any_place(self.AC_COOR_GLOBAL, cand)
            if any_place:
                country, city_name, plat, plon = any_place
                if _same_name(city_name, country):
                    dlat, dlon = self.country_default_coords.get(country, (None, None))
                    return {"Country": country, "City": None, "Latitude": dlat, "Longitude": dlon}
                return {"Country": country, "City": city_name, "Latitude": plat, "Longitude": plon}

            # Try fallback source
            any_place = self._find_any_place(self.AC_V2_GLOBAL, cand)
            if any_place:
                country, city_name, plat, plon = any_place
                if _same_name(city_name, country):
                    dlat, dlon = self.country_default_coords.get(country, (None, None))
                    return {"Country": country, "City": None, "Latitude": dlat, "Longitude": dlon}
                return {"Country": country, "City": city_name, "Latitude": plat, "Longitude": plon}

        # Nothing found
        return {
            "Country": UNKNOWN_COUNTRY,
            "City": None,
            "Latitude": None,
            "Longitude": None
        }

# ==========================
# Helper functions
# ==========================
def make_unique_sheet_name(name: str, used: set[str]) -> str:
    if not isinstance(name, str) or not name.strip():
        name = UNKNOWN_COUNTRY
    s = INVALID_SHEET_CHARS_RE.sub("_", name).strip().strip("'")
    if not s:
        s = "Sheet"
    base = s[:31]
    candidate = base
    i = 1
    while candidate in used:
        suffix = f" ({i})"
        candidate = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(candidate)
    return candidate

def _tokenize_norm(s: str) -> List[str]:
    return [t for t in TOKEN_SPLIT_RE.split(s) if t]

def _candidate_phrases(original: str, max_n: int = 3) -> List[str]:
    """
    Produce normalized candidate phrases (2-grams to 3-grams).
    Build from both original and camelCase-split versions.
    """
    bases = []
    norm1 = _normalize_text(original)
    bases.append(norm1)
    
    split = _maybe_camel_split(original)
    if split:
        norm2 = _normalize_text(split)
        if norm2 != norm1:
            bases.append(norm2)

    seen: set[str] = set()
    phrases: List[str] = []
    
    for base in bases:
        toks = _tokenize_norm(base)
        n = len(toks)
        # Generate 2-grams through max_n-grams
        for size in range(2, min(max_n, n) + 1):
            for i in range(n - size + 1):
                ph = " ".join(toks[i:i+size])
                if ph and ph not in seen:
                    seen.add(ph)
                    phrases.append(ph)

    # Prefer longer phrases first
    phrases.sort(key=len, reverse=True)
    return phrases

def _exact_phrase_in_country(ac: Optional[ahocorasick.Automaton], phrase_norm: str
                             ) -> Optional[Tuple[str, Optional[float], Optional[float]]]:
    """
    Try to match a phrase EXACTLY (the matched key covers the whole phrase).
    """
    if not ac or not phrase_norm:
        return None
    best = None
    best_len = -1
    for end_pos, payload in ac.iter(phrase_norm):
        display_name, plat, plon, klen, is_country_equiv = payload
        if is_country_equiv:
            continue
        start = end_pos - klen + 1
        # Exact match: start at beginning and end at phrase end
        if start == 0 and end_pos == len(phrase_norm) - 1:
            if klen > best_len:
                best_len = klen
                best = (display_name, plat, plon)
    return best

_WORD_CACHE: dict[str, re.Pattern] = {}

# Pre-compile common words
_COMMON_WORDS = ["Sur", "France", "Lebanon", "Pennsylvania"]
for word in _COMMON_WORDS:
    key = _normalize_text(word)
    _WORD_CACHE[key] = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)")

def _has_word(norm_text: str, word: str) -> bool:
    """Case/Unicode-normalized whole-word check."""
    key = _normalize_text(word)
    pat = _WORD_CACHE.get(key)
    if pat is None:
        pat = re.compile(rf"(?<!\w){re.escape(key)}(?!\w)")
        _WORD_CACHE[key] = pat
    return bool(pat.search(norm_text))

# ==========================
# Batch Processing
# ==========================
def detect_batch(unique_list: List[str], detector: CountryCityDetector, batch_size: int = 500) -> List[Dict[str, Any]]:
    """Process detection in batches with progress tracking."""
    results = []
    total = len(unique_list)
    
    for i in range(0, total, batch_size):
        batch = unique_list[i:i+batch_size]
        batch_results = [detector.detect(c) for c in batch]
        results.extend(batch_results)
        
        processed = min(i + batch_size, total)
        print(f"Processed {processed}/{total} ({100*processed/total:.1f}%)")
    
    return results

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    print("=" * 60)
    print("Country-City Detection with Full Variant Support")
    print("=" * 60)
    
    # Read and aggregate data
    t_read = time.perf_counter()
    print("\n[1/4] Reading parquet file...")
    df_counts = (
        pl.scan_parquet("data/ESM_ALL(Countries).parquet")
          .select("SRC_Country")
          .group_by("SRC_Country")
          .len()
          .rename({"len": "count"})
          .collect()
    )
    print(f"✓ Completed in {time.perf_counter() - t_read:.3f} s")

    # Initialize detector
    t_init = time.perf_counter()
    print("\n[2/4] Initializing detector...")
    detector = CountryCityDetector(
        cities_file_v2="data/main_json.json",
        country_names_file="data/country_names.json",
        cities_coor_file="data/main_cities_coor.json",
    )
    print(f"✓ Completed in {time.perf_counter() - t_init:.3f} s")

    # Get unique countries and detect
    t_detect = time.perf_counter()
    unique_countries = df_counts.select("SRC_Country").unique()
    unique_list = unique_countries.get_column("SRC_Country").to_list()
    
    print(f"\n[3/4] Processing {len(unique_list)} unique countries...")
    results = detect_batch(unique_list, detector, batch_size=500)
    
    # Convert results to DataFrame
    df_geo = pl.DataFrame({
        "SRC_Country": unique_list,
        "Country": [r["Country"] for r in results],
        "City": [r["City"] for r in results],
        "Latitude": [r["Latitude"] for r in results],
        "Longitude": [r["Longitude"] for r in results],
    })

    df_classified = df_counts.join(df_geo, on="SRC_Country", how="left")
    print(f"✓ Completed in {time.perf_counter() - t_detect:.3f} s")

    # Ensure Country is not empty/null
    df_classified = df_classified.with_columns(
        pl.when(
            pl.col("Country").is_null() |
            (pl.col("Country").cast(pl.Utf8).str.len_chars().fill_null(0) == 0)
        )
        .then(pl.lit(UNKNOWN_COUNTRY))
        .otherwise(pl.col("Country"))
        .alias("Country")
    )

    # Write Excel output
    t_excel = time.perf_counter()
    print("\n[4/4] Writing Excel file...")
    output_file = "data/grouped_country.xlsx"
    countries = (
        df_classified.select("Country").unique().to_series().to_list()
    )
    countries = [c if isinstance(c, str) and c.strip() else UNKNOWN_COUNTRY for c in countries]

    used_sheet_names: set[str] = set()
    with Workbook(output_file) as wb:
        for idx, country in enumerate(countries):
            safe_name = make_unique_sheet_name(country, used_sheet_names)
            df_filtered = (
                df_classified
                  .filter(pl.col("Country") == country)
                  .select(["SRC_Country", "City", "Latitude", "Longitude", "count"])
                  .sort(["count", "SRC_Country"], descending=[True, False])
            )
            df_filtered.write_excel(
                workbook=wb,
                worksheet=safe_name,
                position=(0, 0),
                table_style="Table Style Medium 4"
            )
            if (idx + 1) % 10 == 0:
                print(f"  Written {idx + 1}/{len(countries)} sheets")

    print(f"✓ Completed in {time.perf_counter() - t_excel:.3f} s")
    print(f"\n{'=' * 60}")
    print(f"SUCCESS! Excel file saved to {output_file}")
    print(f"{'=' * 60}")