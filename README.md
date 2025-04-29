# Candidate Resume Search Web App

This project ingests a dataset of resumes, indexes them with LlamaIndex and Pinecone vector store, and provides a Streamlit web app to search and explore candidate profiles.

---

## Prerequisites

- Python 3.7+
- Pinecone account and API key
- OpenAI API key
- Kaggle API credentials (for dataset download)
  - Install required packages:
  pip install -r requirements.txt


---

## Dataset Download

We use the [resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) from Kaggle.

To download the dataset programmatically, use the Kaggle API

Make sure you have your Kaggle API credentials configured as per [Kaggle API documentation](https://www.kaggle.com/docs/api).

---

## Ingestion Script (`ingestion.py`)

This script:

1. Loads a sample of PDF resumes from the downloaded dataset folder.
2. Parses each PDF into smaller text chunks (nodes) with metadata.
3. Embeds each chunk using OpenAI embeddings.
4. Stores embeddings in Pinecone vector store.
5. Saves the original chunks and metadata locally in a docstore (`./storage` folder) for retrieval.

**Usage:**
    python ingest.py


Open the URL shown in the terminal (usually http://localhost:8501) to interact with the app.

---

## Environment Variables

Set the following environment variables (e.g., in a `.env` file):
- OPENAI_API_KEY=your_openai_api_key
- PINECONE_API_KEY=your_pinecone_api_key


---

## Notes

- Ensure your Pinecone index is created with the correct dimension (1536 for OpenAI embeddings).
- The ingestion script samples 25 PDFs randomly; adjust as needed.
- The app uses caching to speed up repeated queries.
- The LLM extraction prompt can be customized for better accuracy.

---

## References

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Kaggle API](https://www.kaggle.com/docs/api)

---

Feel free to open issues or contribute!

---

