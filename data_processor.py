from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_database(pdf_path):
    print("Loading PDF.")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")

    print("Slicing pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )

    chunks = text_splitter.split_documents(pages)
    print(f"Created {len(chunks)} text chunks.")
    
    print("Downloading free open-source embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Translating {len(chunks)} chunks into vectors and saving to FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    db_folder_name = "thesis_faiss_index"
    vectorstore.save_local(db_folder_name)
    print(f"Success! Database saved to the '{db_folder_name}' folder.")

if __name__ == "__main__":
    build_vector_database("Final_Thesis_KIT.pdf")