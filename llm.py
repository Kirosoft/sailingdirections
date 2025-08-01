import json
import re
from config import openai_client

def llm_extract_features(text: str) -> list[dict]:
    """
    Use LLM to extract features with DMS coords, returning [{name, location:{lat,lon}}].
    Graceful fallback if parse errors.
    """
    system_msg = (
        "You are a geospatial parser. Extract all geographic features and their DDM coordinates "
        "DDM (Degrees and Decimal Minutes) coordinate string, validates it,"
        "The expected format is 'DDMMmmH DDDMMmmH', where:"
        "- DD/DDD: Degrees (2 digits for Lat, 2 or 3 for Lon)"
        "- MM: Minutes"
        "- mm: Decimal part of minutes (hundredths)"
        "- H: Hemisphere ('N', 'S', 'E', 'W')"

        "Example: '441782N 681870W' -> (44.297, -68.311666...)in the form HHMMSSN HHMMSSW as a JSON array of objects {name: string, coords: string}."
        "A compass prefix before a feature name should be combined to form a new feature name e.g."
        "N of Bold Island -> 'N of Bold Island' would be the feature name."
        "Southern Mark Island Ledge -> 'Southern Mark Island Ledge' would be the feature name."
        "Moose Peak Light (white tower, 17 m in height) (442847N 673192W) -> Moose Peak Light (white tower, 17 m in height) would be the feature name."
        "SW extremity of Great Wass Island (442900N 673550W) -> SW extremity of Great Wass Island is the feature name."
        "Fisherman Island (442685N 673660W) and Browney Island (442772N 673712W) -> 'Fisherman Island' and 'Browney Island' are the feature names."
        "W side of the bay Black Rock (442625N 674275W) -> 'W side of the bay Black Rock' is the feature name."
        "or through Tibbett Narrows (442960N 674243W) -> 'Tibbett Narrows' is the feature name."
        "between Petit Manan Point (442370N 675398W) and Dyer Point (442471N 675593W) -> 'Petit Manan Point' , 'Dyer Point' are the feature names."
        "between Cranberry Point (442312N 675900W) and Spruce Point (442141N 680165W) -> 'Cranberry Point' , 'Spruce Point' are the feature names."
        "Sally Island (442406N 675677W) and Sheep Island (442385N 675731W) -> 'Sally Island' , 'Sheep Island' are the feature names."
        "between Dye r Point (2.21) and Youngs Point (442397N 675759W) -> 'Dye r Point' , 'Youngs Point' are the feature names."
        " Do not use 'and' in feature names. Do not include features that do not have valid coordinates."
    )
    usr_msg = f"Extract features from text:\n{text}"
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_msg},
                      {"role":"user","content":usr_msg}],
            temperature=0
        )
        content = resp.choices[0].message.content
        content = re.sub(r'```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        end = content.rfind(']')
        if end!=-1: content = content[:end+1]
        items = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[Warning] JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"[Warning] LLM extraction failed: {e}")
        return []
    feats = []
    for it in items:
        name = it.get("name"); coords=it.get("coords")
        if not name or not coords or coords == '': continue
        try:
            lat, lon = parse_ddm_coordinates(coords)
            feats.append({"name":name, "location":{"lat":lat,"lon":lon}})
        except Exception as e:
            print(f"[Warning] Bad DMS coords '{coords}' in feature '{name}', skipping {e}")
            continue
    return feats


def parse_ddm_coordinates(coord_string: str, feature_name: str = "N/A") -> tuple[float, float] | None:
    """
    Parses a DDM (Degrees and Decimal Minutes) coordinate string, validates it,
    and converts it to decimal degrees.

    The expected format is 'DDMMmmH DDDMMmmH', where:
    - DD/DDD: Degrees (2 digits for Lat, 2 or 3 for Lon)
    - MM: Minutes
    - mm: Decimal part of minutes (hundredths)
    - H: Hemisphere ('N', 'S', 'E', 'W')

    Example: '441782N 681870W' -> (44.297, -68.311666...)

    Args:
        coord_string: The DDM coordinate string.
        feature_name: An optional name of the feature for more descriptive error messages.

    Returns:
        A tuple containing (latitude, longitude) as floats if parsing is successful.
        Returns None if the input is invalid or out of range.
    """
    if not isinstance(coord_string, str):
        print(f"[Warning] Invalid input for feature '{feature_name}': Input must be a string.")
        return None

    parts = coord_string.strip().upper().split()
    if len(parts) != 2:
        print(f"[Warning] Bad DDM coords '{coord_string}' in feature '{feature_name}', skipping. Expected 2 parts, got {len(parts)}.")
        return None

    lat_str, lon_str = parts

    try:
        # --- Latitude Parsing ---
        # Pattern: (DD)(MMmm)(H) -> e.g., (44)(1782)(N)
        lat_match = re.match(r'^(\d{2})(\d{4})([NS])$', lat_str)
        if not lat_match:
            raise ValueError(f"Latitude part '{lat_str}' does not match expected DDM format.")

        lat_deg = int(lat_match.group(1))
        lat_min_decimal = float(f"{lat_match.group(2)[:2]}.{lat_match.group(2)[2:]}")
        lat_hemisphere = lat_match.group(3)

        latitude = lat_deg + (lat_min_decimal / 60.0)
        if lat_hemisphere == 'S':
            latitude *= -1

        # --- Longitude Parsing ---
        # Pattern: (DDD)(MMmm)(H) -> e.g., (068)(1870)(W) or (68)(1870)(W)
        # We'll pad the string to handle both 2 and 3 digit degrees gracefully
        lon_padded_str = lon_str.zfill(8) # e.g. 681870W -> 0681870W
        lon_match = re.match(r'^(\d{3})(\d{4})([EW])$', lon_padded_str)

        if not lon_match:
             raise ValueError(f"Longitude part '{lon_str}' does not match expected DDM format.")

        lon_deg = int(lon_match.group(1))
        lon_min_decimal = float(f"{lon_match.group(2)[:2]}.{lon_match.group(2)[2:]}")
        lon_hemisphere = lon_match.group(3)

        longitude = lon_deg + (lon_min_decimal / 60.0)
        if lon_hemisphere == 'W':
            longitude *= -1

        # --- Validation ---
        if not -90 <= latitude <= 90:
            print(f"[Warning] Bad DDM coords '{coord_string}' in feature '{feature_name}', skipping. Latitude out of range: {latitude:.8f}.")
            return None

        if not -180 <= longitude <= 180:
            print(f"[Warning] Bad DDM coords '{coord_string}' in feature '{feature_name}', skipping. Longitude out of range: {longitude:.8f}.")
            return None

        return (latitude, longitude)

    except (ValueError, TypeError) as e:
        print(f"[Warning] Bad DDM coords '{coord_string}' in feature '{feature_name}', skipping. Error: {e}.")
        return None
    
def parse_dms_pair(s: str) -> tuple[float,float]:
    """
    Convert UKHO DDM ('DDMM.mmN DDDMM.mmW') or DMS ('DDMMSSN DDDMMSSW') to (lat, lon).
    Auto-detects format by length of minute field.
    """
    s = s.strip()
    # Always treat compact UKHO format as DDM: DDMM.mmN DDDMM.mmW or DDMM.mmN DDMM.mmW
    # Try DDM with or without decimal point: DDMM.mmN DDDMM.mmW or DDMMmmN DDDMMmmW
    # UKHO DDM: always 2 digits for lat deg, 2 for lon deg, 2+ for minutes (with or without decimal)
    m = re.match(r"^(\d{2})(\d{2,}(?:\.\d*)?)([NS])\s*(\d{2})(\d{2,}(?:\.\d*)?)([EW])$", s)
    if m:
        d0, m0, dir0, l0, m1, dir1 = m.groups()
        # If no decimal, treat as hundredths of a minute (e.g., 2205 = 22.05')
        if '.' not in m0 and len(m0) > 2:
            m0 = str(int(m0[:2])) + '.' + m0[2:]
        if '.' not in m1 and len(m1) > 2:
            m1 = str(int(m1[:2])) + '.' + m1[2:]
        lat = (int(d0) + float(m0)/60) * (1 if dir0 == 'N' else -1)
        lon = (int(l0) + float(m1)/60) * (1 if dir1 == 'E' else -1)
        # Validate latitude and longitude ranges
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude out of range: {lat} from '{s}'")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude out of range: {lon} from '{s}'")
        return lat, lon
    raise ValueError(f"Bad coordinate format: {s}")

