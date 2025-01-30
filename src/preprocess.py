from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.bedrock import Bedrock
from llama_index.core.schema import TextNode
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.llms import ChatMessage, TextBlock
import copy
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

def create_contextual_nodes(nodes, whole_document, llm):
    """Create contextual nodes with document-level context"""
    nodes_modified = []
    
    prompt_document = """<document>
    {WHOLE_DOCUMENT}
    </document>"""

    prompt_chunk = """Here is the chunk we want to situate within the whole document
    <chunk>
    {CHUNK_CONTENT}
    </chunk>
    Please give a short succinct context to situate this chunk within the overall document for improving search retrieval. Answer only with the succinct context."""
    
    for node in nodes:
        try:
            new_node = copy.deepcopy(node)
            messages = [
                ChatMessage(role="system", content="You are a helpful AI Assistant."),
                ChatMessage(
                    role="user",
                    content=[
                        TextBlock(text=prompt_document.format(WHOLE_DOCUMENT=whole_document)),
                        TextBlock(text=prompt_chunk.format(CHUNK_CONTENT=node.text)),
                    ],
                ),
            ]
            context = str(llm.chat(messages))
            new_node.metadata["context"] = context
            nodes_modified.append(new_node)
            logger.info(f"Created context for node: {new_node.node_id[:8]}...")
        except Exception as e:
            logger.error(f"Error creating context for node: {e}")
            
    return nodes_modified

def main():
    try:
        # Initialize Bedrock LLM
        bedrock_llm = Bedrock(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name="ap-northeast-1"
        )

        # Set up embedding model and settings
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
        Settings.llm = bedrock_llm
        Settings.embed_model = embed_model
        Settings.chunk_size = 1024

        logger.info("Reading documents...")
        reader = SimpleDirectoryReader(
    input_dir="/work/chatbot/data", 
    required_exts=[".md"],
    recursive=True,
)
        docs = reader.load_data()
        logger.info(f"Found {len(docs)} documents")

        # Get raw text and create chunks
        raw_text = "\n\n".join([doc.text for doc in docs])
        
        logger.info("Creating chunks...")
        node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            include_metadata=True,
            include_prev_next_rel=True
        )
        
        nodes = node_parser.get_nodes_from_documents([
            Document(text=raw_text)
        ])
        logger.info(f"Created {len(nodes)} nodes")

        logger.info("Creating contextual nodes...")
        contextual_nodes = create_contextual_nodes(nodes, raw_text, bedrock_llm)

        logger.info("Setting up Qdrant...")
        client = QdrantClient(host="localhost", port=6333)
        vector_store = QdrantVectorStore(
            client=client, 
            collection_name="ubuntu_documents",  
            enable_hybrid=True,
            batch_size=20
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info("Creating index...")
        index = VectorStoreIndex(
            contextual_nodes,
            storage_context=storage_context,
        )

        logger.info("Indexing completed successfully!")

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()