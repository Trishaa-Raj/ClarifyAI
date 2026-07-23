"""
ClarifyAI — Ask questions about any PDF in plain English.
Reads your document, finds the relevant parts, and explains them
at whatever level you need — from beginner to expert.
"""

import os
import re
import fitz          # reads PDF files (PyMuPDF)
import faiss         # fast vector search
import json
import uuid
import time
import logging
import asyncio
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ============================================================
# CONFIG
# ============================================================

# Your OpenRouter API key, loaded from .env file
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set in .env file.")

# OpenRouter uses the same interface as OpenAI — just a different URL
client = OpenAI(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

logging.basicConfig(level=logging.INFO)

# FIX: Max upload size = 20MB. Prevents huge files from crashing the server.
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(title="ClarifyAI")

# Serve the static folder (CSS, JS, images) at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the frontend when someone opens the app in browser
@app.get("/")
async def serve_frontend():
    return FileResponse("templates/index.html")

# Allow requests from any origin (needed for browser to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# EMBEDDING MODEL
# Converts text into numbers (vectors) so we can search by meaning.
# Runs locally on your machine — no API call needed for this.
# Downloads ~90MB on first run, then cached forever.
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# AI MODEL FALLBACK LIST
# ClarifyAI tries these models in order.
# If one fails or hits rate limits, it moves to the next.
# All are free on OpenRouter.
# ============================================================

MODELS = [
    "openai/gpt-oss-120b:free",               # best quality, most thorough
    "meta-llama/llama-3.3-70b-instruct:free", # fast, reliable
    "openai/gpt-oss-20b:free",                # lighter fallback
    "meta-llama/llama-3.1-8b-instruct:free",  # last resort, always available
]

MAX_RETRIES_PER_MODEL = 2

# ============================================================
# GLOBAL DOCUMENT STORE
# Holds the current document's data in memory.
# One document at a time — uploading a new PDF replaces the old one.
# ============================================================

faiss_index = None    # stores text vectors for semantic (meaning-based) search
bm25 = None           # stores text for keyword-based search
chunk_store = []      # the actual text of each chunk
metadata_store = []   # page number and section for each chunk

# ============================================================
# PDF PROCESSING
# Reads the PDF page by page, detects sections, returns structured data.
# ============================================================

def extract_pdf_with_metadata(file_path: str) -> List[Dict]:
    """
    Opens the PDF and extracts text from every page.
    Tries to detect section headings (e.g. INTRODUCTION, METHODS).
    Returns a list of dicts: { text, page, section }
    """
    doc = fitz.open(file_path)
    documents = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if not text.strip():
            # Skip blank pages (e.g. image-only pages)
            continue
        sections = detect_sections(text)
        for section_title, section_text in sections:
            if section_text.strip():
                documents.append({
                    "text": section_text,
                    "page": page_num + 1,
                    "section": section_title
                })

    doc.close()
    return documents


def detect_sections(text: str) -> List[tuple]:
    """
    Splits a page's text by ALL-CAPS headings (e.g. ABSTRACT, RESULTS).
    Falls back to treating the whole page as one 'GENERAL' section
    if no headings are found.
    """
    section_patterns = r"\n([A-Z][A-Z\s]{3,})\n"
    splits = re.split(section_patterns, text)

    sections = []
    for i in range(1, len(splits), 2):
        title = splits[i].strip()
        content = splits[i + 1].strip() if i + 1 < len(splits) else ""
        sections.append((title, content))

    if not sections:
        sections.append(("GENERAL", text))

    return sections

# ============================================================
# SEMANTIC CHUNKING
# Breaks long sections into smaller pieces (~300 chars each).
# Smaller chunks = more precise search results.
# ============================================================

def semantic_chunking(text: str, max_chars: int = 300) -> List[str]:
    """
    Splits text into sentence-aware chunks.
    Tries not to cut mid-sentence by splitting on . ! ?
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# ============================================================
# INDEX BUILDING
# Takes all text chunks and builds two search indexes:
# 1. FAISS — for semantic (meaning-based) search
# 2. BM25  — for keyword (exact word) search
# Both together = hybrid search = better results
# ============================================================

def build_indexes(documents: List[Dict]) -> None:
    """
    Encodes all text chunks into vectors using the local embedding model.
    Stores them in FAISS for fast nearest-neighbour search.
    Also builds a BM25 keyword index for exact word matching.
    """
    global faiss_index, bm25, chunk_store, metadata_store

    chunk_store = []
    metadata_store = []

    for doc in documents:
        chunks = semantic_chunking(doc["text"])
        for chunk in chunks:
            chunk_store.append(chunk)
            metadata_store.append({
                "page": doc["page"],
                "section": doc["section"]
            })

    if not chunk_store:
        raise ValueError("No readable text found in this PDF. It may be a scanned image.")

    # Convert all text chunks to vectors
    embeddings = embedding_model.encode(chunk_store, show_progress_bar=False)
    dim = embeddings.shape[1]

    # Build FAISS index (L2 = Euclidean distance between vectors)
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings))

    # Build BM25 keyword index
    tokenized = [chunk.split() for chunk in chunk_store]
    bm25 = BM25Okapi(tokenized)

# ============================================================
# HYBRID SEARCH
# When you ask a question, this finds the most relevant chunks.
# Combines semantic search (understands meaning) +
# keyword search (finds exact words) for best coverage.
# ============================================================

def hybrid_search(query: str, k: int = 5) -> List[Dict]:
    """
    Runs the question through both FAISS and BM25.
    Merges results — keeping unique chunks from both.
    Guards against out-of-range index values from FAISS.
    """
    if faiss_index is None or bm25 is None:
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please upload and process a PDF first."
        )

    # Semantic search: find chunks whose meaning is closest to the question
    query_vec = embedding_model.encode([query])
    D, I = faiss_index.search(np.array(query_vec), k)
    semantic_results = [int(i) for i in I[0] if i >= 0]

    # Keyword search: find chunks with the most matching words
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    keyword_results = np.argsort(bm25_scores)[::-1][:k].tolist()

    # Merge both result sets, remove duplicates
    combined = list(set(semantic_results) | set(keyword_results))

    results = []
    for idx in combined:
        # Guard: FAISS can return -1 for empty slots
        if 0 <= idx < len(chunk_store):
            results.append({
                "text": chunk_store[idx],
                "metadata": metadata_store[idx]
            })

    return results

# ============================================================
# PROMPT BUILDER
# Constructs the instruction sent to the AI model.
# Includes: the relevant document chunks, the question,
# the explanation level, and the response mode.
# ============================================================

def build_prompt(context_chunks: List[Dict], question: str, mode: str, level: str) -> str:
    """
    Builds the full prompt string sent to the LLM.
    Caps context at 3000 chars total to stay within free model token limits.
    """
    # FIX: Limit total context size to avoid exceeding model token limits
    context_parts = []
    total_chars = 0
    CONTEXT_CHAR_LIMIT = 3000

    for c in context_chunks:
        entry = f"(Page {c['metadata']['page']} | {c['metadata']['section']})\n{c['text']}"
        if total_chars + len(entry) > CONTEXT_CHAR_LIMIT:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    context_text = "\n\n".join(context_parts)

    base_instruction = """
You are ClarifyAI, a document assistant.
Use ONLY the provided context to answer.
If the answer is not in the document, say: "Not found in the document."
Always cite the page number when referencing specific content.
Return your response as valid JSON only — no extra text, no markdown fences.
"""

    level_lower = level.lower()
    if level_lower in ["10 year old", "child", "beginner"]:
        level_instruction = """
Use very simple words a child can understand.
No jargon. Short sentences. Use fun analogies.
"""
    elif level_lower in ["college student", "undergraduate", "student"]:
        level_instruction = """
Use clear language with moderate technical depth.
Define technical terms when you use them.
"""
    elif level_lower in ["researcher", "expert", "phd"]:
        level_instruction = """
Use precise academic language.
Include technical details, equations if present, assumptions and limitations.
"""
    else:
        level_instruction = "Explain clearly and appropriately for a general audience."

    if mode == "equation":
        task = "Focus on equations: explain every variable and what the equation means."
    elif mode == "analysis":
        task = "Analyse the document: cover main argument, methodology, strengths, and weaknesses."
    else:
        task = "Answer the question directly and thoroughly using the context."

    return f"""
{base_instruction}

EXPLANATION LEVEL:
{level_instruction}

TASK:
{task}

DOCUMENT CONTEXT:
{context_text}

QUESTION:
{question}

Respond with this exact JSON structure:
{{
  "main_idea": "The core answer in 1-2 sentences",
  "key_concepts": [{{"concept": "term", "explanation": "what it means"}}],
  "equations_explained": "Explain any equations, or say N/A",
  "real_world_example": "A concrete example from real life, or N/A",
  "simple_summary": "One sentence summary a non-expert can remember"
}}
"""

# ============================================================
# MULTI-MODEL FALLBACK
# Tries each model in MODELS list.
# If a model fails (rate limit, 404, empty response),
# waits briefly and tries the next one.
# FIX: Uses asyncio.sleep instead of time.sleep
# so the server stays responsive during retries.
# ============================================================

async def ask_with_fallback(prompt: str) -> str:
    """
    Sends the prompt to the AI. Tries multiple free models in order.
    Returns the model's text response, or a JSON error if all fail.
    """
    last_error = None

    for model_name in MODELS:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                logging.info(f"ClarifyAI → Trying {model_name} | Attempt {attempt + 1}")

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1500,
                )

                content = response.choices[0].message.content

                if content and content.strip():
                    logging.info(f"ClarifyAI → Success with {model_name}")
                    return content

                raise Exception("Model returned empty response")

            except Exception as e:
                last_error = str(e)
                logging.warning(f"ClarifyAI → {model_name} failed: {e}")
                await asyncio.sleep(1.5)  # FIX: async sleep — server stays responsive

        logging.info(f"ClarifyAI → Moving to next model after {model_name}")

    logging.error("ClarifyAI → All models failed.")
    return json.dumps({
        "error": "All models unavailable",
        "details": last_error
    })

# ============================================================
# REQUEST MODEL
# Defines what the /ask endpoint expects in the request body.
# ============================================================

class Query(BaseModel):
    question: str
    level: str = "college student"   # default explanation level
    mode: str = "normal"             # normal | equation | analysis

# ============================================================
# API ROUTES
# Two endpoints:
# POST /upload — receives a PDF, processes it, builds indexes
# POST /ask    — receives a question, searches, returns AI answer
# ============================================================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF upload.
    FIX 1: Validates it's actually a PDF (checks filename + magic bytes).
    FIX 2: Rejects files over 20MB before processing.
    Extracts text, builds search indexes, returns success message.
    """

    # FIX: Validate file extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file."
        )

    # Read file into memory first
    file_bytes = await file.read()

    # FIX: Validate file size
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        )

    # FIX: Validate PDF magic bytes (real PDFs start with %PDF-)
    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid PDF."
        )

    # Write to a temp file for PyMuPDF to read
    file_id = str(uuid.uuid4())
    file_path = f"temp_{file_id}.pdf"

    try:
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        documents = extract_pdf_with_metadata(file_path)

        if not documents:
            raise HTTPException(
                status_code=422,
                detail="No readable text found in this PDF. It may be a scanned image-only document."
            )

        build_indexes(documents)

    except HTTPException:
        raise  # re-raise clean HTTP errors as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Always delete the temp file, even if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)

    total_chunks = len(chunk_store)
    return {
        "message": f"PDF processed successfully. Indexed {total_chunks} text chunks ready for questions."
    }


@app.post("/ask")
async def ask_question(query: Query):
    """
    Accepts a question about the loaded document.
    Searches for relevant chunks using hybrid search.
    Sends them to the AI with the question and returns a structured answer.
    """
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Find the most relevant chunks from the document
    results = hybrid_search(query.question)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant content found for this question.")

    # Build the prompt and send to AI
    prompt = build_prompt(results, query.question, query.mode, query.level)
    answer = await ask_with_fallback(prompt)

    # Clean up response: strip markdown code fences if model added them
    cleaned = answer.strip()
    cleaned = re.sub(
        r"^```json\s*|^```\s*|```$", "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()

    # Parse JSON response
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # If model didn't return valid JSON, wrap raw text as fallback
        parsed = {"raw_response": cleaned}

    return {"answer": parsed}


# ============================================================
# RUN SERVER (only when running directly, not via uvicorn CLI)
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
