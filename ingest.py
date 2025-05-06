# ingest.py
# Self-contained script to parse nautical sections, extract geo-features, embed content, and index into Elasticsearch with semantic, geo, lexical, and hybrid search support.

import os
import re
import json
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI

# ——— Load env & configure clients —————————————————————————————
load_dotenv()

# Elasticsearch config
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "nautical_sections")
ES_CLOUD_ID   = os.getenv("ES_CLOUD_ID")    # Elastic Cloud ID
ES_API_KEY    = os.getenv("ES_API_KEY")     # single Base64 id:secret API key

# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-ada-002")

# Instantiate clients
es = Elasticsearch(cloud_id=ES_CLOUD_ID, api_key=ES_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ——— Helpers ————————————————————————————————————————————————

def parse_and_chunk(text: str) -> list[dict]:
    """
    Splits Markdown text into sections based on '2.x' headings, capturing chapter/section context.
    """
    sections = []
    parents  = {"chapter": None, "section": None, "subsection": None}
    buffer, current_id, current_title = [], None, None

    for line in text.splitlines():
        if m := re.match(r'^#\s+CHAPTER\s+\d+\s*-\s*(.+)$', line):
            parents["chapter"] = m.group(1)
            continue
        if m := re.match(r'^##\s+(.+)$', line):
            parents["section"] = m.group(1)
            continue
        if m := re.match(r'^###\s+(.+)$', line):
            parents["subsection"] = m.group(1)
            continue
        if m := re.match(r'^(\d+\.\d+)\s+(.+)$', line):
            if buffer and current_id:
                sections.append({
                    "id": current_id,
                    "title": current_title,
                    "parents": {k: v for k, v in parents.items() if v},
                    "content": "\n".join(buffer).strip()
                })
                buffer = []
            current_id, current_title = m.group(1), m.group(2)
            continue
        if current_id and line.strip():
            buffer.append(line)

    if buffer and current_id:
        sections.append({
            "id": current_id,
            "title": current_title,
            "parents": {k: v for k, v in parents.items() if v},
            "content": "\n".join(buffer).strip()
        })
    return sections


def embed(text: str) -> list[float]:
    """Generate embedding using OpenAI 1.x SDK"""
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def llm_extract_features(text: str) -> list[dict]:
    """
    Uses an LLM to extract geographic features and coordinates from text,
    returning a list of {"name": ..., "location": {"lat":.., "lon":..}}.
    Falls back gracefully if the LLM response can't be parsed.
    """
    system_msg = (
        "You are a geospatial parser. Extract all geographic features and their DMS coordinates "
        "in the form HHMMSSN HHMMSSW as a JSON array of objects {name: string, coords: string}."
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
        " Do not use 'and' in feature names"
    )
    user_msg = f"Extract features from the following text:{text}"
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            temperature=0
        )
        content = resp.choices[0].message.content
        # strip markdown fences if present
        content = re.sub(r'```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        items = json.loads(content)
    except json.JSONDecodeError as e:
        # could not parse JSON, log and fallback
        print(f"[Warning] JSON parse error in llm_extract_features: {e}")
        return []
    except Exception as e:
        # any other error from API
        print(f"[Warning] LLM extraction failed: {e}")
        return []

    feats = []
    for item in items:
        coords = item.get("coords")
        name = item.get("name")
        if not coords or not name:
            continue
        try:
            lat, lon = parse_dms_pair(coords)
            feats.append({"name": name, "location": {"lat": lat, "lon": lon}})
        except Exception:
            # skip invalid coords
            continue
    return feats

def extract_features(text: str) -> list[dict]:
    """
    Extract named features and coordinates in DMS 'HHMMSSN HHMMSSW' format.
    Prefers named patterns 'Name (DMS)' or 'Name DMS'; falls back to coords-only.
    """
    feats, seen = [], set()
    named_re = re.compile(
        r"([A-Za-z][A-Za-z0-9\s'\.\-]+?)\s*\(?\s*(\d{2})(\d{2})(\d{2})([NS])\s+"
        r"(\d{2})(\d{2})(\d{2})([EW])\s*\)?"
    )
    for m in named_re.finditer(text):
        name   = m.group(1).strip()
        d0,d1,d2,dir0 = m.group(2),m.group(3),m.group(4),m.group(5)
        l0,l1,l2,dir1 = m.group(6),m.group(7),m.group(8),m.group(9)
        lat = (int(d0) + int(d1)/60 + int(d2)/3600) * (1 if dir0=='N' else -1)
        lon = (int(l0) + int(l1)/60 + int(l2)/3600) * (1 if dir1=='E' else -1)
        seen.add((lat, lon))
        feats.append({"name": name, "location": {"lat": lat, "lon": lon}})

    coord_re = re.compile(r"\b(\d{2})(\d{2})(\d{2})([NS])\s+(\d{2})(\d{2})(\d{2})([EW])\b")
    for d0,d1,d2,dd,l0,l1,l2,ld in coord_re.findall(text):
        lat = (int(d0) + int(d1)/60 + int(d2)/3600) * (1 if dd=='N' else -1)
        lon = (int(l0) + int(l1)/60 + int(l2)/3600) * (1 if ld=='E' else -1)
        if (lat, lon) in seen:
            continue
        name = f"{d0}°{d1}'{d2}\"{dd}, {l0}°{l1}'{l2}\"{ld}"
        feats.append({"name": name, "location": {"lat": lat, "lon": lon}})
    return feats


def parse_dms_pair(dms_str: str) -> tuple[float, float]:
    """
    Parse a DMS string like '442050N 681800W' into (lat, lon).
    """
    m = re.match(r"^(\d{2})(\d{2})(\d{2})([NS])\s*(\d{2})(\d{2})(\d{2})([EW])$", dms_str.strip())
    if not m:
        raise ValueError(f"Invalid DMS coordinate pair: '{dms_str}'")
    d0,d1,d2,dir0,l0,l1,l2,dir1 = m.groups()
    lat = (int(d0) + int(d1)/60 + int(d2)/3600) * (1 if dir0=='N' else -1)
    lon = (int(l0) + int(l1)/60 + int(l2)/3600) * (1 if dir1=='E' else -1)
    return lat, lon

# ——— Index setup ————————————————————————————————————————————————

def ensure_index(es: Elasticsearch, name: str):
    """
    Ensure the Elasticsearch index exists with the correct mapping, and update mapping if needed.
    """
    mapping_body = {
        "properties": {
            "section_id":     {"type": "keyword"},
            "title":          {"type": "text"},
            "parents":        {"type": "object"},
            "content":        {"type": "text"},
            "content_vector": {"type": "dense_vector", "dims": 1536},
            "features": {
                "type": "nested",
                "properties": {
                    "name":     {"type": "text"},
                    "location": {"type": "geo_point"}
                }
            },
            "locations":     {"type": "geo_point"},
            "feature_names": {"type": "keyword"}
        }
    }
    if not es.indices.exists(index=name):
        es.indices.create(index=name, body={"mappings": mapping_body})
    else:
        es.indices.put_mapping(index=name, body=mapping_body)


# ——— Ingestion ————————————————————————————————————————————————

def index_sections(es: Elasticsearch, sections: list[dict], index_name: str):
    for sec in sections:
        vec  = embed(sec["content"])
        # Use LLM-based extraction instead of regex
        feats = llm_extract_features(sec["content"])
        locations = [feat["location"] for feat in feats]
        doc  = {
            "section_id": sec["id"],
            "title":      sec["title"],
            "parents":    sec["parents"],
            "content":    sec["content"],
            "content_vector": vec,
            "features":   feats,
            "locations": locations,
            # flatten feature names for kibana tooltip
            "feature_names": [feat['name'] for feat in feats]
        }
        es.index(index=index_name, id=sec["id"], body=doc)
        print(f"Upserted section {sec['id']} with {len(feats)} features")

# ——— Semantic search —————————————————————————————————————————————

def semantic_search(es: Elasticsearch, index_name: str, query: str, k: int = 5) -> list[dict]:
    qv = embed(query)
    body = {
        "size": k,
        "query": {"script_score": {
            "query": {"match_all": {}},
            "script": {"source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0", "params": {"query_vector": qv}}
        }}
    }
    res = es.search(index=index_name, body=body)
    return [{"section_id": h['_source']['section_id'], "title": h['_source']['title'], "score": h['_score']} for h in res['hits']['hits']]

# ——— Geo & DMS search —————————————————————————————————————————————

def geo_search(es: Elasticsearch, index_name: str, lat: float, lon: float, distance: str = "10km") -> list[dict]:
    body = {"query": {"nested": {"path": "features", "query": {"geo_distance": {"distance": distance, "features.location": {"lat": lat, "lon": lon}}}}}}
    res = es.search(index=index_name, body=body)
    return [{"section_id": h['_source']['section_id'], "features": h['_source']['features'], "score": h['_score']} for h in res['hits']['hits']]

def geo_search_dms(es: Elasticsearch, index_name: str, coord_input: str, distance: str = "10km") -> list[dict]:
    if isinstance(coord_input, str):
        lat, lon = parse_dms_pair(coord_input)
    else:
        lat, lon = coord_input
    return geo_search(es, index_name, lat, lon, distance)

# ——— LLM-based feature extraction is now used in ingestion

# ——— Lexical search —————————————————————————————————————————————

def lexical_search(es: Elasticsearch, index_name: str, term: str) -> list[dict]:
    body = {"query": {"bool": {"should": [
        {"match_phrase": {"content": term}},
        {"nested": {"path": "features", "query": {"match_phrase": {"features.name": term}}}}
    ]}}}
    res = es.search(index=index_name, body=body)
    results = []
    for h in res['hits']['hits']:
        src = h['_source']
        matched = []
        if term.lower() in src['content'].lower(): matched.append('content')
        if any(term.lower() in f['name'].lower() for f in src['features']): matched.append('features')
        results.append({"section_id": src['section_id'], "title": src['title'], "matched_in": matched, "score": h['_score']})
    return results

# ——— Hybrid search —————————————————————————————————————————————

def hybrid_search(es: Elasticsearch, index_name: str, query: str, alpha: float = 0.5, k: int = 5) -> list[dict]:
    qv = embed(query)
    body = {
        "size": k,
        "query": {"script_score": {
            "query": {"bool": {"should": [
                {"match_phrase": {"content": {"query": query, "boost": 1-alpha}}},
                {"nested": {"path": "features", "query": {"match_phrase": {"features.name": {"query": query, "boost": 1-alpha}}}}}
            ]}},
            "script": {"source": "(cosineSimilarity(params.query_vector, 'content_vector') + 1.0) * params.alpha + _score * (1 - params.alpha)", "params": {"query_vector": qv, "alpha": alpha}}
        }}
    }
    res = es.search(index=index_name, body=body)
    return [{"section_id": h['_source']['section_id'], "title": h['_source']['title'], "score": h['_score']} for h in res['hits']['hits']]

# ——— Main entrypoint —————————————————————————————————————————————

if __name__ == "__main__":
    with open("e-NP68_17_2021-chapter2.md", encoding="utf-8") as f:
        raw = f.read()

    secs = parse_and_chunk(raw)
    ensure_index(es, ES_INDEX_NAME)
    index_sections(es, secs, ES_INDEX_NAME)

    print("\n-- Semantic Search Example --")
    for r in semantic_search(es, ES_INDEX_NAME, "Bar Island", k=3):
        print(f"[{r['section_id']}] {r['title']} (score {r['score']:.3f})")

    print("\n-- Geo Search Example (DMS) --")
    for r in geo_search_dms(es, ES_INDEX_NAME, "442390N 681240W", "1m"):
        print(f"Section {r['section_id']} with features {r['features']}")

    print("\n-- Lexical Search Example --")
    for r in lexical_search(es, ES_INDEX_NAME, "Bar Island"):
        print(f"[{r['section_id']}] {r['title']} matched in {r['matched_in']} (score {r['score']:.3f})")

    print("\n-- Hybrid Search Example --")
    for r in hybrid_search(es, ES_INDEX_NAME, "Bar Island", alpha=0.7, k=5):
        print(f"[{r['section_id']}] {r['title']} (score {r['score']:.3f})")
