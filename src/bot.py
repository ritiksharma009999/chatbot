from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.llms.bedrock import Bedrock
from llama_index.core.query_engine import RetrieverQueryEngine
from qdrant_client import QdrantClient
from typing import List, Optional
import os
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    similarity_top_k: Optional[int] = 10
    sparse_top_k: Optional[int] = 12

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]




import boto3
import logging

@app.on_event("startup")
async def startup_event():
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='ap-northeast-1',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        logging.info("Successfully initialized Bedrock client")
    except Exception as e:
        logging.error(f"Error initializing Bedrock client: {e}")
        raise

class HybridContextualRetriever(BaseRetriever):
    """Custom retriever that combines vector, sparse, and ColBERT reranking"""
    
    def __init__(
        self,
        vector_store_index: VectorStoreIndex,
        similarity_top_k: int = 2,
        sparse_top_k: int = 12,
        reranker: ColbertRerank = None,
    ):
        self.index = vector_store_index
        self.similarity_top_k = similarity_top_k
        self.sparse_top_k = sparse_top_k
        self.reranker = reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        try:
            retriever = self.index.as_retriever(
                similarity_top_k=self.similarity_top_k,
                sparse_top_k=self.sparse_top_k,
                vector_store_query_mode="hybrid"
            )
            
            nodes = retriever.retrieve(query_bundle)
            
            if self.reranker is not None:
                nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
                for node in nodes:
                    node.node.metadata["retrieval_score"] = float(node.score)
            
            return nodes
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            raise

def initialize_search_engine():
    """Initialize the search engine components"""
    try:
        # Initialize Bedrock LLM
        bedrock_llm = Bedrock(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name="ap-northeast-1"
        )

        # Set up embedding model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        Settings.llm = bedrock_llm
        Settings.embed_model = embed_model

        # Set up Qdrant

        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        

        vector_store = QdrantVectorStore(
            client=client, 
            collection_name="ubuntu_docs",
            enable_hybrid=True
        )

        # Load the existing index
        index = VectorStoreIndex.from_vector_store(vector_store)

        # Initialize ColBERT reranker
        colbert_reranker = ColbertRerank(
            top_n=10,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            keep_retrieval_score=True,
        )

        return index, colbert_reranker

    except Exception as e:
        logger.error(f"Error initializing search engine: {e}")
        raise

# Initialize components at startup
index, colbert_reranker = initialize_search_engine()

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        # Create retriever
        retriever = HybridContextualRetriever(
            vector_store_index=index,
            similarity_top_k=request.similarity_top_k,
            sparse_top_k=request.sparse_top_k,
            reranker=colbert_reranker
        )

        # Create query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[colbert_reranker],
            response_mode="compact"
        )

        # Execute query
        response = query_engine.query(request.query)

        # Prepare sources
        sources = []
        for node in response.source_nodes:
            sources.append({
                "node_id": node.node.node_id,
                "content": node.node.get_content()[:150],
                "colbert_score": node.score,
                "retrieval_score": node.node.metadata.get("retrieval_score", "N/A")
            })

        return QueryResponse(
            answer=str(response),
            sources=sources
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)