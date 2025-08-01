# ingest.py
# Self-contained script to parse nautical sections, extract geo-features via LLM, embed content, and index into two Elasticsearch indices with search utilities.

import re
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI


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






