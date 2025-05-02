import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
import json
import re

load_dotenv()

@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    st.info("Loading index...")

    # Initialize OpenAI LLM and embeddings
    Settings.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        embed_batch_size=100,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Initialize PGVector vector store
    vector_store = PGVectorStore.from_params(
        host="localhost",
        port=5432,
        database="vectordb",
        user="llama",
        password="llama_pw",
        table_name="resume_embeddings",
        embed_dim=1536,
    )

    # Load storage context with persisted docstore and vector store
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir="./storage",  # Must match ingestion persist_dir
    )

    # Load index with hydrated docstore
    index = load_index_from_storage(storage_context)
    return index

@st.cache_data(show_spinner=False)
def cached_extract_with_llm(text: str):
    if not isinstance(text, str) or not text.strip():
        return {"name": "Unknown", "profession": "Unknown", "years_experience": 0}

    prompt = f"""
You are a helpful assistant that extracts structured information from resumes.

Extract the following fields from the resume text below:
- Full Name (if present)
- Profession (the main job title or current/most recent role)
- Years of commercial experience (sum up all relevant professional experience, estimate if necessary)

Resume text:
{text}

Respond in JSON with keys: name, profession, years_experience. If a field is missing, use "Unknown" or 0.
"""
    try:
        response = Settings.llm.complete(prompt)
        # st.write("LLM raw response:", response)  # Log raw response for debugging

        # Extract the 'text' field from the response if it's a dict-like string
        # If response is a string, try to parse as JSON first
        if isinstance(response, str):
            try:
                response_json = json.loads(response)
                llm_text = response_json.get("text", response)
            except Exception:
                llm_text = response
        elif isinstance(response, dict):
            llm_text = response.get("text", "")
        else:
            llm_text = str(response)

        # Now parse the actual JSON string inside llm_text
        try:
            data = json.loads(llm_text)
        except Exception:
            # Try to extract JSON substring from llm_text
            match = re.search(r'\{.*\}', llm_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = {"name": "Unknown", "profession": "Unknown", "years_experience": 0}

        # Normalize years_experience to float
        years_exp = data.get("years_experience", 0)
        if isinstance(years_exp, str):
            try:
                years_exp = float(years_exp)
            except Exception:
                years_exp = 0
        data["years_experience"] = years_exp

        return data

    except Exception as e:
        st.error(f"LLM extraction error: {e}")
        return {"name": "Unknown", "profession": "Unknown", "years_experience": 0}

def get_all_candidates(index: VectorStoreIndex, max_candidates: int = 25):
    all_nodes = list(index.docstore.docs.values())
    print(f"Retrieved {len(all_nodes)} nodes from index")

    # Show metadata of first few nodes for debugging
    for node in all_nodes[:3]:
        print("Node metadata:", node.metadata)
        print("Node text snippet:", node.text[:200])

    # Group chunks by source_file metadata (filename)
    grouped = {}
    for node in all_nodes:
        source = node.metadata.get("filename", "unknown")
        grouped.setdefault(source, []).append(
            node.get_content() if hasattr(node, "get_content") else node.text
        )

    candidates = []
    for source, chunk_texts in list(grouped.items())[:max_candidates]:
        full_resume = "\n".join(chunk_texts)
        full_resume = full_resume[:4000]  # truncate to avoid too long prompt
        info = cached_extract_with_llm(full_resume)
        candidates.append({
            "Name": source,
            "Profession": info.get("profession", "Unknown"),
            "Years Experience": info.get("years_experience", 0),
            "Resume Preview": full_resume  # Store the full resume text here
        })
    return candidates


def main():
    st.title("Candidate Resumes")

    index = get_index()
    candidates = get_all_candidates(index)

    if not candidates:
        st.info("No candidates found in the database.")
    else:
        for c in candidates:
            # Use filename as header
            st.subheader(f"Filename: {c['Name']}")
            st.write(f"**Profession:** {c['Profession']}")
            st.write(f"**Years of Experience:** {c['Years Experience']}")
            # Create an expander for detailed info
            with st.expander("Show detailed info and summary"):

                # Show full resume text or summary here
                # Assuming you want to show more detailed text, you can add it to candidates dict
                # For now, let's show the full resume preview text stored in 'Resume Preview'
                st.write(c['Resume Preview'])

            st.markdown("---")

if __name__ == "__main__":
    main()
