import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq # <-- The New Engine!

# Load your API keys
load_dotenv()

def ask_chatbot(user_question):
    print("Waking up the AI...")
    
    # Load the local database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("thesis_faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Connect to Groq's servers
    print("Connecting to Groq...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant", # Using Meta's powerful Llama 3 model
        temperature=0.2,
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If the answer is not in the context, say 'I cannot answer this based on the provided document.' "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\nThinking...\n")
    response = rag_chain.invoke({"input": user_question})
    
    return response["answer"]

if __name__ == "__main__":
    question = "How can I make habit attactive if I find it hard to follow?"
    answer = ask_chatbot(question)
    
    print("--- AI ANSWER ---")
    print(answer)