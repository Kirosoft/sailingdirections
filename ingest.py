# ingest.py
# Self-contained script to parse nautical sections, extract geo-features via LLM, embed content, and index into two Elasticsearch indices with search utilities.

import os
import re
import json
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI

# ——— Load env & configure clients —————————————————————————————
load_dotenv()

# Elasticsearch config
ES_INDEX_NAME     = os.getenv("ES_INDEX_NAME", "nautical_sections")
ES_FEATURES_INDEX = os.getenv("ES_FEATURES_INDEX", "nautical_features")
ES_CLOUD_ID       = os.getenv("ES_CLOUD_ID")    # Elastic Cloud ID
ES_API_KEY        = os.getenv("ES_API_KEY")     # single Base64 id:secret API key

# OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-ada-002")

# Instantiate clients
es = Elasticsearch(cloud_id=ES_CLOUD_ID, api_key=ES_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ——— Helpers ————————————————————————————————————————————————

def parse_and_chunk(text: str) -> list[dict]:
    """Split Markdown into numbered sections, capturing headings as metadata."""
    sections = []
    parents = {"chapter": None, "section": None, "subsection": None}
    buf, cur_id, cur_title = [], None, None
    for line in text.splitlines():
        if m := re.match(r'^#\s+CHAPTER\s+\d+\s*-\s*(.+)$', line):
            parents["chapter"] = m.group(1); continue
        if m := re.match(r'^##\s+(.+)$', line):
            parents["section"] = m.group(1); continue
        if m := re.match(r'^###\s+(.+)$', line):
            parents["subsection"] = m.group(1); continue
        if m := re.match(r'^(\d+\.\d+)\s+(.+)$', line):
            if buf and cur_id:
                sections.append({"id":cur_id, "title":cur_title,
                                 "parents":{k:v for k,v in parents.items() if v},
                                 "content":"\n".join(buf).strip()})
                buf = []
            cur_id, cur_title = m.group(1), m.group(2)
            continue
        if cur_id and line.strip(): buf.append(line)
    if buf and cur_id:
        sections.append({"id":cur_id, "title":cur_title,
                         "parents":{k:v for k,v in parents.items() if v},
                         "content":"\n".join(buf).strip()})
    return sections


def embed(text: str) -> list[float]:
    """Generate embedding via OpenAI 1.x"""
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def llm_extract_features(text: str) -> list[dict]:
    """
    Use LLM to extract features with DMS coords, returning [{name, location:{lat,lon}}].
    Graceful fallback if parse errors.
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
        if not name or not coords: continue
        try:
            lat, lon = parse_dms_pair(coords)
            feats.append({"name":name, "location":{"lat":lat,"lon":lon}})
        except:
            continue
    return feats


def parse_dms_pair(s: str) -> tuple[float,float]:
    """Convert 'DDMMSSN DDDMMSSW' to (lat,lon)."""
    m = re.match(r"^(\d{2})(\d{2})(\d{2})([NS])\s*(\d{2})(\d{2})(\d{2})([EW])$", s.strip())
    if not m: raise ValueError(f"Bad DMS: {s}")
    d0,d1,d2,dir0,l0,l1,l2,dir1 = m.groups()
    lat = (int(d0)+int(d1)/60+int(d2)/3600)*(1 if dir0=='N' else -1)
    lon = (int(l0)+int(l1)/60+int(l2)/3600)*(1 if dir1=='E' else -1)
    return lat, lon

# ——— Index setup ————————————————————————————————————————————————

def ensure_index(es,name:str):
    mapping={"properties":{
        "section_id":{"type":"keyword"},"title":{"type":"text"},
        "parents":{"type":"object"},"content":{"type":"text"},
        "content_vector":{"type":"dense_vector","dims":1536},
        "features":{"type":"nested","properties":{
            "name":{"type":"text"},"location":{"type":"geo_point"}}},
        "locations":{"type":"geo_point"},
        "feature_names":{"type":"keyword"}
    }}
    if not es.indices.exists(index=name):
        es.indices.create(index=name,body={"mappings":mapping})
    else:
        es.indices.put_mapping(index=name,body=mapping)


def ensure_features_index(es,name:str):
    mapping={"properties":{
        "feature_id":{"type":"keyword"},
        "name":{"type":"text"},
        "location":{"type":"geo_point"},
        "section_id":{"type":"keyword"}
    }}
    if not es.indices.exists(index=name):
        es.indices.create(index=name,body={"mappings":mapping})
    else:
        es.indices.put_mapping(index=name,body=mapping)

# ——— Ingestion ————————————————————————————————————————————————

def index_sections(es,sections,index_name,features_index):
    for sec in sections:
        vec = embed(sec["content"])
        feats = llm_extract_features(sec["content"])
        locs = [f["location"] for f in feats]
        doc={
            "section_id":sec["id"],"title":sec["title"],
            "parents":sec["parents"],"content":sec["content"],
            "content_vector":vec,"features":feats,
            "locations":locs,"feature_names":[f['name'] for f in feats]
        }
        es.index(index=index_name,id=sec["id"],body=doc)
        print(f"Upserted {sec['id']} with {len(feats)} feats")
        seen=set()
        for feat in feats:
            key=(feat['name'],feat['location']['lat'],feat['location']['lon'],sec['id'])
            if key in seen: continue
            seen.add(key)
            fid=f"{sec['id']}_{feat['name'].replace(' ','_')}"
            fdoc={"feature_id":fid,"name":feat['name'],"location":feat['location'],"section_id":sec['id']}
            es.index(index=features_index,id=fid,body=fdoc)

# ——— Searches —————————————————————————————————————————————

def semantic_search(es,index_name,query,k=5):
    qv=embed(query)
    body={"size":k,"query":{"script_score":{
        "query":{"match_all":{}},
        "script":{"source":"cosineSimilarity(params.query_vector,'content_vector')+1.0","params":{"query_vector":qv}}
    }}}
    res=es.search(index=index_name,body=body)
    return [{"section_id":h['_source']['section_id'],"title":h['_source']['title'],"score":h['_score']} for h in res['hits']['hits']]


def geo_search(es,index_name,lat,lon,distance="10km"):
    body={"query":{"nested":{"path":"features","query":{"geo_distance":{"distance":distance,"features.location":{"lat":lat,"lon":lon}}}}}}
    res=es.search(index=index_name,body=body)
    return [{"section_id":h['_source']['section_id'],"features":h['_source']['features'],"score":h['_score']} for h in res['hits']['hits']]


def geo_search_dms(es,coord_input,distance="10km"):
    if isinstance(coord_input,str): lat,lon=parse_dms_pair(coord_input)
    else: lat,lon=coord_input
    body={"query":{"geo_distance":{"distance":distance,"location":{"lat":lat,"lon":lon}}}}
    res=es.search(index=ES_FEATURES_INDEX,body=body)
    return [{"feature_id":h['_source']['feature_id'],"name":h['_source']['name'],"location":h['_source']['location'],"section_id":h['_source']['section_id'],"score":h['_score']} for h in res['hits']['hits']]


def lexical_search(es,index_name,term):
    body={"query":{"bool":{"should":[
        {"match_phrase":{"content":term}},
        {"nested":{"path":"features","query":{"match_phrase":{"features.name":term}}}}
    ]}}}
    res=es.search(index=index_name,body=body)
    out=[]
    for h in res['hits']['hits']:
        src=h['_source']; m=[]
        if term.lower() in src['content'].lower(): m.append('content')
        if any(term.lower() in f['name'].lower() for f in src['features']): m.append('features')
        out.append({"section_id":src['section_id'],"title":src['title'],"matched_in":m,"score":h['_score']})
    return out


def hybrid_search(es,index_name,query,alpha=0.5,k=5):
    qv=embed(query)
    body={"size":k,"query":{"script_score":{
        "query":{"bool":{"should":[
            {"match_phrase":{"content":{"query":query,"boost":1-alpha}}},
            {"nested":{"path":"features","query":{"match_phrase":{"features.name":{"query":query,"boost":1-alpha}}}}}
        ]}},
        "script":{"source":"(cosineSimilarity(params.query_vector,'content_vector')+1.0)*params.alpha+_score*(1-params.alpha)","params":{"query_vector":qv,"alpha":alpha}}
    }}}
    res=es.search(index=index_name,body=body)
    return [{"section_id":h['_source']['section_id'],"title":h['_source']['title'],"score":h['_score']} for h in res['hits']['hits']]

# ——— Main —————————————————————————————————————————————
if __name__=="__main__":
    with open("e-NP68_17_2021-chapter2.md",encoding="utf-8") as f: raw=f.read()
    secs=parse_and_chunk(raw)
    ensure_index(es,ES_INDEX_NAME)
    ensure_features_index(es,ES_FEATURES_INDEX)
    index_sections(es,secs,ES_INDEX_NAME,ES_FEATURES_INDEX)
    print("\n-- Semantic Search --")
    for r in semantic_search(es,ES_INDEX_NAME,"Bar Island",k=3): print(r)
    print("\n-- Geo Search (DMS) --")
    for r in geo_search_dms(es,"442390N 681240W","1m"): print(r)
    print("\n-- Lexical Search --")
    for r in lexical_search(es,ES_INDEX_NAME,"Bar Island"): print(r)
    print("\n-- Hybrid Search --")
    for r in hybrid_search(es,ES_INDEX_NAME,"Bar Island",alpha=0.7,k=5): print(r)
