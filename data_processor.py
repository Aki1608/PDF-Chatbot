from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(pdf_path):
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
    
    return chunks

if __name__ == "__main__":
    print("Getting chunks.")
    my_chunks = process_pdf("Habit.pdf")

    if my_chunks:
        print("####### First chunk of the PDF. #########")
        print(my_chunks[0].page_content)