import gradio as gr
from qa_engine import ask_chatbot

def response_to_user(question):
    bot_answer = ask_chatbot(question)
    return bot_answer

demo = gr.ChatInterface(
    fn=response_to_user,
    title="Thesis AI assistant.",
    description="Ask any question about the PDF (My thesis). This bot uses local FAISS embeddings and Groq's Llama 3 to find the answer.",
    theme="soft",
    examples=[
        "What is the main conclusion of this thesis?",
        "What is the main topic of this document? Can you give me a sort summary of whole document?"
    ]

)

if __name__ == "__main__":
    demo.launch()
