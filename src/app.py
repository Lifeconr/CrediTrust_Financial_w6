import gradio as gr
import os, sys
from src.rag_pipeline import RAGPipeline

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize RAG pipeline
VECTOR_STORE_PATH = 'vector_store/'
rag_pipeline = RAGPipeline(
    model_name="google/gemma-2b-it",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    vector_store_path=VECTOR_STORE_PATH
)

# Load components
rag_pipeline.load_embedding_model()
rag_pipeline.load_faiss_index()
rag_pipeline.load_llm()
rag_pipeline.setup_rag_chain()

# Gradio interface function
def query_rag(question: str):
    if not question:
        return "Please enter a question.", "No sources available."
    try:
        answer, _, retrieved_docs = rag_pipeline.query(question)
        sources = "\n".join([f"- {doc['text_content'][:100]}..." for doc in retrieved_docs])
        return answer, sources
    except Exception as e:
        return f"Error: {str(e)}", "No sources available due to error."

# Clear interface function
def clear_fields():
    return "", "", ""

# Build Gradio interface
with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown("# CrediTrust Complaint Analysis Assistant")
    gr.Markdown("Ask about customer complaints across financial products (e.g., 'What issues are reported with BNPL?').")
    
    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(label="Question", placeholder="e.g., Why are customers unhappy with credit cards?")
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")
        with gr.Column(scale=2):
            answer_output = gr.Markdown(label="Answer")
            sources_output = gr.Markdown(label="Sources")
    
    submit_btn.click(query_rag, inputs=question_input, outputs=[answer_output, sources_output])
    clear_btn.click(clear_fields, outputs=[question_input, answer_output, sources_output])

if __name__ == "__main__":
    demo.launch(inbrowser=True, show_api=False)