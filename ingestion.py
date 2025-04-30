import os
import random
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.storage.vector_store.pinecone import PineconeVectorStore
from llama_index.readers.file import UnstructuredReader

from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = PineconeGRPC(api_key=PINECONE_API_KEY)

index_name = "llamaindex-resumes"
dimension = 1536

# Create Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=dimension,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Setup OpenAI embedding and LLM
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

Settings.embed_model = embed_model
Settings.llm = llm

folder_path = "./llamaindex-resumes/data/data/ENGINEERING"

def get_random_pdf_files(folder_path, sample_size=25):
    all_pdfs = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder_path, f))
    ]
    return random.sample(all_pdfs, sample_size) if len(all_pdfs) > sample_size else all_pdfs

random_pdfs = get_random_pdf_files(folder_path, sample_size=25)

# Load documents with metadata
reader = SimpleDirectoryReader(
    input_files=random_pdfs,
    file_extractor={".pdf": UnstructuredReader()}
)
documents = reader.load_data()

# Parse documents into nodes with source_file metadata
parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)
nodes = []
for doc in documents:
    doc_nodes = parser.get_nodes_from_documents([doc])
    for node in doc_nodes:
        node.metadata["source_file"] = os.path.basename(doc.metadata.get("file_path", "unknown"))
    nodes.extend(doc_nodes)

# Embed nodes
for node in nodes:
    node.embedding = embed_model.get_text_embedding(node.text)


# Create a docstore and add nodes
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

# Create storage context with your vector store and docstore
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    docstore=docstore
)

# Build index from nodes and storage context
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

# Persist storage context (saves nodes metadata locally)
storage_context.persist(persist_dir="./storage")

print(f"Ingested {len(nodes)} chunks from {len(random_pdfs)} PDF files.")
print(pinecone_index.describe_index_stats())
