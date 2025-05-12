import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Union, Dict
from pydantic import BaseModel, Field

from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

import json
import re

load_dotenv()

# -------------------------------
# Pydantic Models
# -------------------------------

class ResumeInfo(BaseModel):
    name: str = Field(default="Unknown")
    profession: str = Field(default="Unknown")
    years_experience: Union[int, float] = Field(default=0)

class Candidate(BaseModel):
    file_name: str
    profession: str
    years_experience: Union[int, float]
    resume_preview: str


# -------------------------------
# Load Index
# -------------------------------

@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    st.info("Loading index...")

    Settings.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        embed_batch_size=100,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vector_store = PGVectorStore.from_params(
        host="localhost",
        port=5432,
        database="vectordb",
        user="llama",
        password="llama_pw",
        table_name="resume_embeddings",
        embed_dim=1536,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir="./storage",
    )

    return load_index_from_storage(storage_context)


# -------------------------------
# LLM-based Resume Extraction
# -------------------------------

@st.cache_data(show_spinner=False)
def cached_extract_with_llm(text: str) -> ResumeInfo:
    if not isinstance(text, str) or not text.strip():
        return ResumeInfo()

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

        # Handle response format variations
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
                llm_text = parsed.get("text", response)
            except json.JSONDecodeError:
                llm_text = response
        elif isinstance(response, dict):
            llm_text = response.get("text", "")
        else:
            llm_text = str(response)

        # Extract JSON content
        try:
            data = json.loads(llm_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", llm_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                return ResumeInfo()

        return ResumeInfo(**data)

    except Exception as e:
        st.error(f"LLM extraction error: {e}")
        return ResumeInfo()


# -------------------------------
# Retrieve All Candidates
# -------------------------------

def get_all_candidates(index: VectorStoreIndex, max_candidates: int = 25) -> List[Candidate]:
    all_nodes = list(index.docstore.docs.values())
    print(f"Retrieved {len(all_nodes)} nodes from index")

    grouped: Dict[str, List[str]] = {}
    for node in all_nodes:
        source = node.metadata.get("filename", "unknown")
        grouped.setdefault(source, []).append(
            node.get_content() if hasattr(node, "get_content") else node.text
        )

    candidates: List[Candidate] = []

    for source, chunks in list(grouped.items())[:max_candidates]:
        full_resume = "\n".join(chunks)[:4000]  # Truncate to avoid too long prompts
        extracted_info = cached_extract_with_llm(full_resume)

        candidate = Candidate(
            file_name=source,
            profession=extracted_info.profession,
            years_experience=extracted_info.years_experience,
            resume_preview=full_resume,
        )
        candidates.append(candidate)

    return candidates


# -------------------------------
# Streamlit App
# -------------------------------

def main():
    st.title("Candidate Resumes")

    index = get_index()
    candidates = get_all_candidates(index)

    if not candidates:
        st.info("No candidates found in the database.")
        return

    for candidate in candidates:
        st.subheader(f"Filename: {candidate.file_name}")
        st.write(f"**Profession:** {candidate.profession}")
        st.write(f"**Years of Experience:** {candidate.years_experience}")

        with st.expander("Show detailed info and summary"):
            st.write(candidate.resume_preview)

        st.markdown("---")


if __name__ == "__main__":
    main()
