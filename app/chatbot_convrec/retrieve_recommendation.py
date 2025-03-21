from app.retrieval.retriever import RetrievalConfig, VectorStoreRetriever
from app.retrieval.query_transformers import LLMQueryTransformer
from llama_index.core.schema import NodeWithScore
from jinja2 import Template
from app.chatbot_convrec.defaults import DEFAULT_CONSTRAINTS_TO_QUERY_TRANSFORMER, DEFAULT_RECOMMENDATION_RETRIEVER


def retrieve_recommendation(hard_constraints: dict,
                            soft_constraints: dict,
                            constraints_to_query_transformer: LLMQueryTransformer = DEFAULT_CONSTRAINTS_TO_QUERY_TRANSFORMER,
                            vec_retriever: VectorStoreRetriever = DEFAULT_RECOMMENDATION_RETRIEVER) -> list[NodeWithScore]:

    query = constraints_to_query_transformer.transform_query(

        {
            "hard_constraints": hard_constraints,
            "soft_constraints": soft_constraints
        }
    )

    return vec_retriever.retrieve(query)
