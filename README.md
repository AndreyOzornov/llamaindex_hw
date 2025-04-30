# Candidate Resume Search Web App

This project ingests a dataset of resumes, indexes them with LlamaIndex and Pinecone vector store, and provides a Streamlit web app to search and explore candidate profiles.

---

## Prerequisites

- Python 3.7+
- Pinecone account and API key
- OpenAI API key
- Kaggle API credentials (for dataset download)
- Required packages (see requirements.txt)

---

# Project Setup

Follow these steps to set up the project environment and install the required dependencies.

## Setup Virtual Environment

1. Create a new virtual environment using Python3:

```bash
python3 -m venv myenv
```

2. Activate the virtual environment:

```bash
source myenv/bin/activate
```

## Environment Variables

Set the following environment variables (in a `.env` file):
- OPENAI_API_KEY=your_openai_api_key
- PINECONE_API_KEY=your_pinecone_api_key

---

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Download

We use the [resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) from Kaggle.

To download the dataset programmatically, use the Kaggle API

Make sure you have your Kaggle API credentials configured as per [Kaggle API documentation](https://www.kaggle.com/docs/api).

  1. Execute the following command to download dataset:  
```bash
python3 download_dataset.py
```
  2. Execute the following command to initialize the database and insert the data:  
```bash
python3 ingestion.py
```
  3. Execute the following command to start the server:  
```bash
python3 app.py
```
  4. Open the following URL in your browser:  
     `http://localhost:8501`

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
