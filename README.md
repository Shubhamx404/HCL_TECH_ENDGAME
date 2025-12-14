# 🧾 Legal Document Analyser using RAG

*A Retrieval-Augmented Generation (RAG) based system to quickly understand and analyse legal documents such as constitutions, acts, and case laws.*

---

## 👥 Contributors

- **Aditya Amarnath** – 22035002 (Ceramic Engg)  
- **Shubham Kumar Sharma** – 22045132 (Chemical Engg)  
- **Lakshya Aryan** – 22035039 (Ceramic Engg)  
- **Devang Darpe** – 22035025 (Ceramic Engg)

---

## 📌 Project Overview

The **Legal Document Analyser** is an AI-powered tool that helps users:

- Upload legal documents (PDFs, books, research papers, etc.)
- Automatically chunk and index them into a vector database
- Ask natural language questions about the content
- Get accurate, context-aware answers with proper legal references
- Generate **case briefs** from long judgments or legal texts

This is built using a **RAG (Retrieval-Augmented Generation)** pipeline, where relevant chunks are first retrieved from a vector database and then passed to an LLM for response generation.

---

## 🧠 Tech Stack

- **Language Models (LLMs)**  
  - Open-source models from **Hugging Face**  
  - Optionally **Gemini models via API** (if available)  

- **Fine-tuning / Optimization (if base system is not performing well)**  
  - **LoRA / QLoRA** based finetuning  
  - Quantization-aware setups for low VRAM environments  

- **Core Components**
  - Document loaders (PDFs, books, research papers)
  - Text chunking utilities
  - Embedding models
  - Vector database for similarity search
  - RAG pipeline for answer generation

- **(Optional) Frontend / Interfaces**
  - Web UI (e.g., Streamlit / Gradio / custom web app)
  - CLI interface for quick queries

---

## 🔍 Key Features

- 📂 **PDF & Document Upload**  
  Upload legal documents such as constitution texts, acts, and research papers.

- ✂️ **Smart Document Chunking**  
  Large documents are split into smaller, meaningful chunks for efficient retrieval.

- 🧬 **Embeddings + Vector DB**  
  - Embeddings are generated for each chunk.  
  - Stored in a vector database for fast similarity search.

- ⚖️ **Legal Case Brief Generator**  
  A dedicated function to generate **case briefs** from long judgments:
  - Facts of the case  
  - Issues  
  - Arguments  
  - Judgment  
  - Conclusion  

- 🔎 **RAG-based Query Handling**  
  - User query → Similar chunks retrieved using **similarity search**  
  - Retrieved context + query passed to LLM  
  - LLM generates a clear, structured legal answer.

- 🚀 **Deployable as**  
  - Web UI  
  - CLI tool  

---

## 🏗️ System Workflow

1. **Upload PDF / Legal Document**  
   - User uploads constitution data, legal acts, or case law PDFs.

2. **Document Loading & Preprocessing**  
   - Use document loaders to read PDFs/books/research papers.  
   - Clean text (remove headers, footers, bad characters, etc.)

3. **Text Chunking**  
   - Long text is split into smaller chunks (e.g., 512–1024 tokens).  
   - Each chunk is associated with metadata (page no., section, etc.)

4. **Embedding Generation**  
   - Load embedding model.  
   - Create a function to generate embeddings for every chunk.

5. **Store in Vector Database**  
   - Store embeddings + metadata in a vector DB.  
   - Enables fast similarity search during query time.

6. **Query Handling using RAG**  
   - User asks a question about the uploaded legal document.  
   - Retrieve **top-k** relevant chunks from vector DB.  
   - Feed query + retrieved chunks to an LLM.  
   - LLM generates a context-aware answer.

7. **Case Brief Generation (Optional)**  
   - For long judgments or cases, call the case-brief function.  
   - Generates a structured summary (facts, issues, judgment, etc.)

8. **Deployment**  
   - Expose the pipeline via:
     - Web UI (e.g. `/app_web.py`)  
     - CLI interface (e.g. `/app_cli.py`)

---
# workflow -> How we implementing rag.
[workflow](![WhatsApp Image 2025-12-04 at 4 37 46 PM](https://github.com/user-attachments/assets/749810e3-5bf1-4acc-988d-068fe81119ab)

-We deployed it using streamlit

# demo working page 
[page](![WhatsApp Image 2025-12-04 at 19 45 00_acd236cc](https://github.com/user-attachments/assets/45c40f73-9f40-4ef8-a07a-f49428d89986)

## 📂 Project Structure (Example)

> **Note:** Adjust filenames to match your actual implementation.

```bash
legal-document-analyser/
├── data/
│   ├── raw_documents/
│   └── processed/
├── embeddings/
│   └── vector_store/          # Serialized vector DB
├── src/
│   ├── config.py              # API keys, model paths, constants
│   ├── document_loader.py     # Functions to load PDFs/books
│   ├── chunking.py            # Text chunking utilities
│   ├── embeddings.py          # Embedding model + generation
│   ├── vector_db.py           # Vector DB init + search
│   ├── rag_pipeline.py        # RAG workflow (retrieve + generate)
│   ├── case_brief.py          # Case brief generation logic
│   ├── cli_app.py             # CLI interface
│   └── web_app.py             # Web UI (Streamlit/Gradio/etc.)
├── notebooks/
│   └── experiments.ipynb      # Prototyping & experiments
├── requirements.txt
├── README.md
└── LICENSE


## Methodology

-load constitution data (books , pdf , reserach paper etc )
-then we do document chunking (documentloder for documentload)
-we create  function for case breif - genrate case breif 
-text chunking - we chunk larger data into smaller chunks (Tokenization)

-we do store  embeddings  using vector database(**A futuristic digital illustration of a legal AI assistant. A glowing digital version of a 
Constitution book is open in the center, with streams of binary code and glowing nodes connecting specific legal articles to a chatbot 
interface on a glass tablet. The background is a clean, professional dark blue with cyber-security aesthetics. High tech, detailed, isometric view.**)

-then load embedding model 
-then we create a function to genrete embedding - then we crete a function to identify the most relevent docment to resopnse 

we use Rag to retreive the most relevent chunk 

#workflow ->
#upload pdf -> processing ->embedding and processing ->strore in vector db ->query handling using rag -> using similarity serch to get most relevent response->deploy (web ui , Cli)

# workflow -> How we implementing rag.
[workflow](![WhatsApp Image 2025-12-04 at 4 37 46 PM](https://github.com/user-attachments/assets/749810e3-5bf1-4acc-988d-068fe81119ab)

-We deployed it using streamlit

# demo working page 
[page](![WhatsApp Image 2025-12-04 at 19 45 00_acd236cc](https://github.com/user-attachments/assets/45c40f73-9f40-4ef8-a07a-f49428d89986)






