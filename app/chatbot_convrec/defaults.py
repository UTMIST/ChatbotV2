
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from app.retrieval.query_transformers import LLMQueryTransformer
from jinja2 import Template
from app.retrieval.retriever import VectorStoreRetriever


from dotenv import load_dotenv
import os


load_dotenv()

VEC_STORE = SimpleVectorStore.from_persist_path("./storage/ml_resources/vec_store.json")

LLM = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
DATA_SOURCE_FOLDER = "./app/data/input"
DATA_SOURCE_FINISHED_FOLDER = "./app/data/finished"
CONSTRAINTS_TO_QUERY_PROMPT_PATH = "./app/prompts/constraints_to_query.jinja"

DEFAULT_CONSTRAINTS_TO_QUERY_TRANSFORMER = LLMQueryTransformer(
    llm=LLM,
    prompt_template=Template(open(CONSTRAINTS_TO_QUERY_PROMPT_PATH).read())
)

DEFAULT_RECOMMENDATION_RETRIEVER = VectorStoreRetriever(
    vector_store=VEC_STORE
)
