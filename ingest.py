import re
from elasticsearch import Elasticsearch
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.DEBUG)

# ——— Configuration ———
ES_CLOUD_ID       = os.getenv("ES_CLOUD_ID")
ES_INDEX_NAME     = os.getenv("ES_INDEX_NAME", "nautical_sections")
ES_API_KEY_SECRET = os.getenv("ES_API_KEY_SECRET")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-ada-002")


# authenticate to Elastic Cloud using API key
es = Elasticsearch(
    cloud_id=ES_CLOUD_ID,
    api_key=ES_API_KEY_SECRET,
)

# smoke-test:
try:
    info = es.info()
    print("✅ Connected to cluster:", info["cluster_name"])
except Exception as e:
    print("❌ Connection failed:", e)
    raise


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_and_chunk(text):
    """
    Walks through the text line by line, tracking the most recent
    chapter / sections (via '#', '##', '###' headings) and then splits
    out each '2.x' section into its own record, capturing the parent
    metadata at that point.
    """
    sections = []
    parents = {"chapter": None, "section": None, "subsection": None}
    buffer = []
    current_id = None
    current_title = None

    for line in text.splitlines():
        # Chapter heading
        if m := re.match(r'^#\s+CHAPTER\s+\d+\s*-\s*(.+)$', line):
            parents["chapter"] = m.group(0)
            continue

        # Level-2 heading
        if m := re.match(r'^##\s+(.+)$', line):
            parents["section"] = m.group(1)
            continue

        # Level-3 heading
        if m := re.match(r'^###\s+(.+)$', line):
            parents["subsection"] = m.group(1)
            continue

        # Numbered section start (e.g. 2.1 Title)
        if m := re.match(r'^(\d+\.\d+)\s+(.+)$', line):
            # save previous section
            if buffer and current_id:
                sections.append({
                    "id": current_id,
                    "title": current_title,
                    "parents": {k: v for k, v in parents.items() if v},
                    "content": "\n".join(buffer).strip()
                })
                buffer = []

            current_id = m.group(1)
            current_title = m.group(2)
            continue

        # accumulate content
        if line.strip() and current_id:
            buffer.append(line)

    # final section
    if buffer and current_id:
        sections.append({
            "id": current_id,
            "title": current_title,
            "parents": {k: v for k, v in parents.items() if v},
            "content": "\n".join(buffer).strip()
        })

    return sections

def ensure_index(es, name):
    """
    Create the index with a dense_vector mapping if it doesn't exist.
    """
    # note: exists() must be called with index=<name>, not a positional arg
    if not es.indices.exists(index=name):
        body = {
            "mappings": {
                "properties": {
                    "section_id":     {"type": "keyword"},
                    "title":          {"type": "text"},
                    "parents":        {"type": "object"},
                    "content":        {"type": "text"},
                    "content_vector": {"type": "dense_vector", "dims": 1536}
                }
            }
        }
        # create also already uses keyword args
        es.indices.create(index=name, body=body)

def embed(text: str) -> list[float]:
    # Note: `.data` is a list of Pydantic models; use `.data[0].embedding`
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]           # can be a single string or list of strings
    )
    return resp.data[0].embedding

def index_sections(es, sections, index_name):
    for sec in sections:
        vec = embed(sec["content"])
        doc = {
            "section_id": sec["id"],
            "title":      sec["title"],
            "parents":    sec["parents"],
            "content":    sec["content"],
            "content_vector": vec
        }
        es.index(index=index_name, id=sec["id"], body=doc)
        print(f"Indexed section {sec['id']}")

def semantic_search(es, index_name, query, k=10):
    # 1) Embed the query text
    q_vec = embed(query)

    # 2) Build & send the k-NN search request
    body = {
        "size": k,
        "query": {
            "script_score": {
                "query": { "match_all": {} },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                    "params": { "query_vector": q_vec }
                }
            }
        }
    }
    res = es.search(index=index_name, body=body)

    # 3) Extract & return the hits
    hits = []
    for hit in res["hits"]["hits"]:
        hits.append({
            "section_id": hit["_source"]["section_id"],
            "title":      hit["_source"]["title"],
            "content":      hit["_source"]["content"],
            "score":      hit["_score"]
        })
    return hits

if __name__ == "__main__":
    skip_load = True

    if not skip_load:
        # Load your raw text however you like:
        with open("e-NP68_17_2021-chapter2.md", encoding="utf-8") as f:
            raw = f.read()

        secs = parse_and_chunk(raw)
        ensure_index(es, ES_INDEX_NAME)
        index_sections(es, secs, ES_INDEX_NAME)
    else:
        ensure_index(es, ES_INDEX_NAME)

        # Assuming you’ve already run ensure_index() and index_sections()
        query = "major lights and buoys"
        results = semantic_search(es, ES_INDEX_NAME, query, k=3)

        print(f"Top 10 sections for query: “{query}”\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['section_id']}] {r['title']}  {r['content']} (score={r['score']:.4f})")
