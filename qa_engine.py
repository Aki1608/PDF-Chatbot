import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def ask_chatbot(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("thesis_faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Create a retriever that fetches the top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Connecting to LLM...")
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta", # <-- THE FIX
        temperature=0.2, 
        max_new_tokens=250
    )

    # 4. The Strict System Prompt (Guardrail against hallucinations)
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

    # 5. Assemble and run the pipeline
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\nThinking...\n")
    response = rag_chain.invoke({"input": user_question})
    
    return response["answer"]

if __name__ == "__main__":
    # Test it out! Replace this with a real question about your Master's thesis.
    question = "What is the main conclusion of this book?"
    answer = ask_chatbot(question)
    
    print("--- AI ANSWER ---")
    print(answer)