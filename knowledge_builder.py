"""Build a local PMC-backed knowledge base for Medical RAG.

This script implements a strict sequential ingestion pipeline:
1) Discover open-access PMC records via E-search.
2) Fetch and parse each article XML via E-fetch.
3) Chunk article text semantically.
4) Embed and store chunks in persistent local ChromaDB.

No async concurrency is used to protect unified-memory systems.
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'langchain-chroma'. Install it with: pip install langchain-chroma"
    ) from exc

from model_setup import ensure_local_ollama_models, release_loaded_connectors

LOGGER = logging.getLogger("knowledge_builder")

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_DELAY_SECONDS = 1.5


@dataclass
class BuilderConfig:
    """Configuration for retrieval, chunking, and vector storage."""

    search_term: str
    max_results: int = 5
    chunk_size: int = 1200
    chunk_overlap: int = 200
    persist_directory: str = "./local_chroma_db"
    collection_name: str = "pmc_medical_literature"
    user_agent: str = "local-medical-rag/0.1 (personal-research-use)"
    ollama_host: Optional[str] = None


def _rate_limited_get(url: str, user_agent: str) -> requests.Response:
    """GET with mandatory NCBI-safe delay after every API call."""
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=60)
    response.raise_for_status()
    # CRITICAL API RULE: must pause between every E-utilities request.
    time.sleep(API_DELAY_SECONDS)
    return response


def search_pmc(query: str, max_results: int = 5, user_agent: str = "local-medical-rag/0.1") -> List[str]:
    """Discover open-access PMC IDs via NCBI E-search."""
    encoded_query = quote(query)
    endpoint = (
        f"{EUTILS_BASE}/esearch.fcgi"
        f"?db=pmc&term={encoded_query}+AND+open+access[filter]&retmode=json&retmax={max_results}"
    )
    LOGGER.info("E-search request for query: %s", query)
    response = _rate_limited_get(endpoint, user_agent=user_agent)

    payload = response.json()
    id_list = payload.get("esearchresult", {}).get("idlist", [])
    pmcids = [str(item).strip() for item in id_list if str(item).strip()]
    LOGGER.info("E-search returned %s PMC IDs", len(pmcids))
    return pmcids


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_and_parse_article(pmcid: str, user_agent: str = "local-medical-rag/0.1") -> Optional[Dict[str, str]]:
    """Fetch article XML via E-fetch and parse title, abstract, and body.

    Returns None when parsing yields no usable text.
    """
    endpoint = f"{EUTILS_BASE}/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml"
    LOGGER.info("E-fetch request for PMCID: %s", pmcid)

    try:
        response = _rate_limited_get(endpoint, user_agent=user_agent)
    except requests.RequestException as exc:
        LOGGER.warning("Failed to fetch PMCID %s: %s", pmcid, exc)
        return None

    soup = BeautifulSoup(response.text, "xml")

    # Remove high-noise sections before extracting text content.
    for tag_name in ["ref-list", "table-wrap"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title_tag = soup.find("article-title")
    abstract_tag = soup.find("abstract")
    body_tag = soup.find("body")

    title = _normalize_whitespace(title_tag.get_text(" ", strip=True)) if title_tag else ""
    abstract = _normalize_whitespace(abstract_tag.get_text(" ", strip=True)) if abstract_tag else ""
    body = _normalize_whitespace(body_tag.get_text(" ", strip=True)) if body_tag else ""

    if not any([title, abstract, body]):
        LOGGER.warning("PMCID %s has no usable title/abstract/body content", pmcid)
        return None

    combined_text = "\n\n".join(
        part for part in [f"Title: {title}" if title else "", f"Abstract: {abstract}" if abstract else "", body] if part
    ).strip()

    return {
        "pmcid": str(pmcid),
        "title": title,
        "abstract": abstract,
        "body": body,
        "text": combined_text,
        "source": "PMC",
    }


def _build_splitter(chunk_size: int = 1200, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _create_vector_store(
    persist_directory: str,
    collection_name: str,
    embedding_model: OllamaEmbeddings,
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )


def _chunk_article(article: Dict[str, str], splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    base_doc = Document(
        page_content=article["text"],
        metadata={
            "pmcid": article["pmcid"],
            "title": article["title"],
            "source": article["source"],
        },
    )
    return splitter.split_documents([base_doc])


def build_knowledge_base(config: BuilderConfig) -> Dict[str, int]:
    """End-to-end build: search -> fetch/parse -> chunk -> sequential embed/store."""
    ensure_local_ollama_models()

    pmcids = search_pmc(
        query=config.search_term,
        max_results=config.max_results,
        user_agent=config.user_agent,
    )
    splitter = _build_splitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=config.ollama_host)
    vector_store = _create_vector_store(
        persist_directory=config.persist_directory,
        collection_name=config.collection_name,
        embedding_model=embeddings,
    )

    parsed_count = 0
    skipped_count = 0
    stored_chunks = 0

    LOGGER.info("Starting sequential fetch/parse/chunk/embed for %s IDs", len(pmcids))

    for index, pmcid in enumerate(pmcids, start=1):
        LOGGER.info("Processing PMCID %s (%s/%s)", pmcid, index, len(pmcids))
        article = fetch_and_parse_article(pmcid=pmcid, user_agent=config.user_agent)
        if article is None:
            skipped_count += 1
            continue

        parsed_count += 1
        chunks = _chunk_article(article, splitter)
        if not chunks:
            skipped_count += 1
            continue

        vector_store.add_documents(chunks)
        stored_chunks += len(chunks)
        LOGGER.info(
            "Embedded/stored PMCID %s with %s chunks (running total: %s)",
            pmcid,
            len(chunks),
            stored_chunks,
        )

        # Sequential memory hygiene between articles.
        del chunks
        gc.collect()

    release_loaded_connectors()
    gc.collect()

    return {
        "fetched_ids": len(pmcids),
        "parsed_articles": parsed_count,
        "skipped_articles": skipped_count,
        "stored_chunks": stored_chunks,
    }


def parse_args() -> BuilderConfig:
    """Parse command line options into BuilderConfig."""
    parser = argparse.ArgumentParser(description="Build local medical literature vector DB from PMC.")
    parser.add_argument("--search-term", required=True, help="PMC search query, e.g. 'cardiac arrhythmia treatments'")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of open-access PMC papers")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for text splitting")
    parser.add_argument("--persist-directory", default="./local_chroma_db", help="Directory for persistent local Chroma")
    parser.add_argument("--collection-name", default="pmc_medical_literature", help="Chroma collection name")
    parser.add_argument("--ollama-host", default=None, help="Optional Ollama host URL")

    args = parser.parse_args()
    return BuilderConfig(
        search_term=args.search_term,
        max_results=args.max_results,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_directory=args.persist_directory,
        collection_name=args.collection_name,
        ollama_host=args.ollama_host,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = parse_args()
    LOGGER.info("Starting local knowledge base build with search term: %s", config.search_term)

    summary = build_knowledge_base(config)
    LOGGER.info("Build complete: %s", summary)


if __name__ == "__main__":
    main()
