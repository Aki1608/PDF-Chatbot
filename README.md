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

## Getting Started

### 1. Prerequisites
* Python 3.10+
* A free [Groq API Key](https://console.groq.com/keys)
* A free [Hugging Face Access Token](https://huggingface.co/settings/tokens)

### 2. Installation & Environment Setup
It is highly recommended to run this project inside a Virtual Environment to avoid dependency conflicts.
```bash
# Clone the repository and navigate into it
git clone <your-repo-url>
cd <your-repo-folder>

# Create and activate a clean virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required architecture
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory of the project. Do not commit this file to version control.
```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_key_here
GROQ_API_KEY=your_groq_key_here
```

### 4. Build the Local Database
Place your target PDF in the root folder (ensure it is named `Final_Thesis_KIT.pdf` or update the script accordingly), and run the processor:
```bash
python data_processor.py
```
*(This will create a hidden folder named `thesis_faiss_index` containing your processed database).*

### 5. Launch the Web App
Start the Gradio server:
```bash
python app.py
```
Click the local URL generated in your terminal (e.g., `http://127.0.0.1:7860`) to start chatting.