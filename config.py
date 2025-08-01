
import os
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

RE_INDEX = os.getenv("RE_INDEX", "false").lower() in ("true", "1", "yes")


# Instantiate clients
es = Elasticsearch(cloud_id=ES_CLOUD_ID, api_key=ES_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)