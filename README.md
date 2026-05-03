# Master's Thesis AI Assistant

A full-stack Retrieval-Augmented Generation (RAG) application that allows users to interactively chat with a Master's thesis. This project uses local vector embeddings for secure document search and leverages Groq's lightning-fast inference engine to generate accurate, context-aware answers.

## Features
* **Interactive Web UI:** Clean, modern chat interface powered by Gradio.
* **Local Vector Database:** Uses FAISS and Hugging Face embeddings (`all-MiniLM-L6-v2`) to process and store document chunks locally, ensuring data privacy.
* **High-Speed LLM:** Integrated with Meta's `llama-3.1-8b-instant` via Groq for near-instant text generation.
* **Modular Architecture:** Clean separation of concerns between the data processor, the RAG engine, and the frontend UI.

## Tech Stack
* **Python 3**
* **LangChain** (Classic, Community, HuggingFace, Groq)
* **FAISS** (Local Vector Store)
* **Gradio** (Frontend UI)
* **Hugging Face Hub** (Embeddings)
* **Groq** (LLM Provider)

## Project Structure
* `app.py` - The Gradio frontend web server and user interface.
* `qa_engine.py` - The backend RAG pipeline connecting the local database to the cloud LLM.
* `data_processor.py` - The ETL script that reads the PDF, slices it into readable chunks, and builds the FAISS vector database.
* `requirements.txt` - The exact list of dependencies needed to run the environment.
* `my_thesis.pdf` - The source document the AI reads from.

## Quick Start Guide

### 1. Set Up the Environment
It is highly recommended to run this inside a Virtual Environment to avoid dependency conflicts.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory of the project and add your API keys for Hugging Face and Groq:
```env
GROQ_API_KEY=your_groq_key_here
```

### 3. Build the Database
Ensure your PDF (`Final_Thesis_KIT.pdf`) is in the main folder, then run the data processor to generate the local vector embeddings:
```bash
python data_processor.py
```
*(This will create a hidden folder named `thesis_faiss_index` containing your processed database).*

### 4. Launch the Application
Start the frontend web server:
```bash
python app.py
```
Click the local URL generated in the terminal (e.g., `http://127.0.0.1:7860`) to open the chat interface in your browser.