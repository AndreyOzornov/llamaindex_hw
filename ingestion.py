import os
import random
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import UnstructuredReader

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = "./storage"
DOCSTORE_FILENAME = "docstore.json"  # default filename used by SimpleDocumentStore
FOLDER_PATH = "./llamaindex-resumes/data/data/ENGINEERING"
SAMPLE_SIZE = 25

# PostgreSQL / pgvector connection params
PG_PARAMS = dict(
    host="localhost",
    port=5432,
    database="vectordb",
    user="llama",
    password="llama_pw",
    table_name="resume_embeddings",
    embed_dim=1536,
)


def get_random_pdf_files(folder_path, sample_size=SAMPLE_SIZE):
    """Return a random sample of PDF file paths from the folder."""
    if not os.path.exists(folder_path):
        raise ValueError(f"Directory {folder_path} does not exist")

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    sample_size = min(sample_size, len(pdf_files))
    sampled_files = random.sample(pdf_files, sample_size)
    return [os.path.join(folder_path, f) for f in sampled_files]


def main():
    # Initialize OpenAI embedding and LLM
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    # Initialize PGVector vector store
    vector_store = PGVectorStore.from_params(**PG_PARAMS)

    # Load or create docstore
    docstore_path = os.path.join(PERSIST_DIR, DOCSTORE_FILENAME)
    if os.path.exists(docstore_path):
        print(f"Loading existing docstore from {docstore_path}")
        docstore = SimpleDocumentStore.from_persist_path(docstore_path)
    else:
        print("Creating new docstore")
        docstore = SimpleDocumentStore()

    # Load documents
    random_pdfs = get_random_pdf_files(FOLDER_PATH, SAMPLE_SIZE)
    print(f"Selected {len(random_pdfs)} PDF files for ingestion.")

    reader = SimpleDirectoryReader(
        input_files=random_pdfs,
        file_extractor={".pdf": UnstructuredReader()},
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents.")

    # Parse documents into nodes
    parser = SimpleNodeParser.from_defaults(
        chunk_size=500,
        chunk_overlap=20,
        include_metadata=True,
    )
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Parsed {len(nodes)} nodes from documents.")

    # Add nodes to docstore (existing + new)
    docstore.add_documents(nodes)
    print(f"Docstore now contains {len(docstore.docs)} documents.")

    # Create storage context with vector store and docstore
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )

    # Build index from nodes and storage context
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    # Persist storage context (saves docstore and metadata locally)
    storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Storage context persisted to {PERSIST_DIR}")

    print(f"Ingested {len(nodes)} chunks from {len(random_pdfs)} PDF files.")


if __name__ == "__main__":
    main()