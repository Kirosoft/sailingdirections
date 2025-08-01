from config import *
from llm import llm_extract_features, parse_dms_pair

# ——— Index setup ————————————————————————————————————————————————
def embed(text: str) -> list[float]:
    """Generate embedding via OpenAI 1.x"""
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

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

