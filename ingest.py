# ingest.py
# Self-contained script to parse nautical sections, extract geo-features, embed content, and index into Elasticsearch with semantic, geo, and lexical search support.

import os
import re
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
    sections, parents = [], {"chapter": None, "section": None, "subsection": None}
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
        name = m.group(1).strip()
        d0, d1, d2, dir0 = m.group(2), m.group(3), m.group(4), m.group(5)
        l0, l1, l2, dir1 = m.group(6), m.group(7), m.group(8), m.group(9)
        lat = (int(d0) + int(d1) / 60 + int(d2) / 3600) * (1 if dir0.upper() == 'N' else -1)
        lon = (int(l0) + int(l1) / 60 + int(l2) / 3600) * (1 if dir1.upper() == 'E' else -1)
        seen.add((lat, lon))
        feats.append({"name": name, "location": {"lat": lat, "lon": lon}})

    coord_re = re.compile(r"\b(\d{2})(\d{2})(\d{2})([NS])\s+(\d{2})(\d{2})(\d{2})([EW])\b")
    for d0, d1, d2, dd, l0, l1, l2, ld in coord_re.findall(text):
        lat = (int(d0) + int(d1) / 60 + int(d2) / 3600) * (1 if dd.upper() == 'N' else -1)
        lon = (int(l0) + int(l1) / 60 + int(l2) / 3600) * (1 if ld.upper() == 'E' else -1)
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
    d0, d1, d2, dir0, l0, l1, l2, dir1 = m.groups()
    lat = (int(d0) + int(d1) / 60 + int(d2) / 3600) * (1 if dir0.upper() == 'N' else -1)
    lon = (int(l0) + int(l1) / 60 + int(l2) / 3600) * (1 if dir1.upper() == 'E' else -1)
    return lat, lon

# ——— Index setup ————————————————————————————————————————————————

def ensure_index(es: Elasticsearch, name: str):
    if not es.indices.exists(index=name):
        es.indices.create(index=name, body={
            "mappings": {"properties": {
                "section_id": {"type": "keyword"},
                "title": {"type": "text"},
                "parents": {"type": "object"},
                "content": {"type": "text"},
                "content_vector": {"type": "dense_vector", "dims": 1536},
                "features": {"type": "nested", "properties": {
                    "name": {"type": "text"},
                    "location": {"type": "geo_point"}
                }}
            }}
        })

# ——— Ingestion ————————————————————————————————————————————————

def index_sections(es: Elasticsearch, sections: list[dict], index_name: str):
    for sec in sections:
        vec = embed(sec["content"])
        feats = extract_features(sec["content"])
        doc = {
            "section_id": sec["id"],
            "title": sec["title"],
            "parents": sec["parents"],
            "content": sec["content"],
            "content_vector": vec,
            "features": feats
        }
        es.index(index=index_name, id=sec["id"], body=doc)
        print(f"Upserted section {sec['id']} with {len(feats)} features")

# ——— Semantic search —————————————————————————————————————————————

def semantic_search(es: Elasticsearch, index_name: str, query: str, k: int = 5) -> list[dict]:
    qv = embed(query)
    body = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {"source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0", "params": {"query_vector": qv}}
            }
        }
    }
    res = es.search(index=index_name, body=body)
    return [{
        "section_id": h['_source']['section_id'],
        "title": h['_source']['title'],
        "score": h['_score']
    } for h in res['hits']['hits']]

# ——— Geo search ———————————————————————————————————————————————

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
    """
    Perform a hybrid search combining semantic vector score and lexical match score.
    `alpha` controls balance: 0 = pure lexical, 1 = pure semantic.
    """
    # 1) embed query
    qv = embed(query)
    body = {
        "size": k,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match_phrase": {"content": {"query": query, "boost": 1-alpha}}},
                            {"nested": {"path": "features", "query": {"match_phrase": {"features.name": {"query": query, "boost": 1-alpha}}}}}
                        ]
                    }
                },
                "script": {
                    "source": "(cosineSimilarity(params.query_vector, 'content_vector') + 1.0) * params.alpha + _score * (1 - params.alpha)",
                    "params": {"query_vector": qv, "alpha": alpha}
                }
            }
        }
    }
    res = es.search(index=index_name, body=body)
    results = []
    for h in res['hits']['hits']:
        results.append({
            "section_id": h['_source']['section_id'],
            "title": h['_source']['title'],
            "score": h['_score']
        })
    return results

# ——— Main entrypoint —————————————————————————————————————————————

if __name__ == "__main__":
    with open("e-NP68_17_2021-chapter2.md", encoding="utf-8") as f:
        raw = f.read()

    secs = parse_and_chunk(raw)
    ensure_index(es, ES_INDEX_NAME)
    #index_sections(es, secs, ES_INDEX_NAME)

    print("\n-- Semantic Search Example --")
    for r in semantic_search(es, ES_INDEX_NAME, "Bar Island", k=3):
        print(f"[{r['section_id']}] {r['title']} (score {r['score']:.3f})")

    print("\n-- Geo Search Example (DMS) --")
    # bar island coords: 441543N 682744W
    for r in geo_search_dms(es, ES_INDEX_NAME, "441543N 682744W", "0.5km"):
        print(f"Section {r['section_id']} with features {r['features']}")

    print("-- Lexical Search Example --")
    for r in lexical_search(es, ES_INDEX_NAME, "Bar Island"):
        print(f"[{r['section_id']}] {r['title']} matched in {r['matched_in']} (score {r['score']:.3f})")

    # Hybrid search example
    print("-- Hybrid Search Example --")
    for r in hybrid_search(es, ES_INDEX_NAME, "Bar Island", alpha=0.7, k=5):
        print(f"[{r['section_id']}] {r['title']} (score {r['score']:.3f})")
