import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Helper function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load the PNG image from the assets folder
image = Image.open("assets/minion.png")

# Set page config as the first Streamlit command
st.set_page_config(page_title="Neo4py QA Bot", page_icon=image, layout="wide")

import faiss
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
import pymupdf4llm
import os
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index" not in st.session_state:
    st.session_state.index = None
if "pdfs_loaded" not in st.session_state:
    st.session_state.pdfs_loaded = False

# System prompt for the chatbot
SYSTEM_PROMPT = """
You are a helpful and knowledgeable assistant specialized in answering questions about Neo4py and related topics. You always analyze the prompt and then answer in Markdown format.
Provide clear, concise, detailed and accurate responses. If you don't know the answer, say so and suggest where the user might find more information.
"""

def read_pdf(file_path):
    try:
        # Use pymupdf4llm with the file path
        md_text = pymupdf4llm.to_markdown(file_path)
        return md_text
    except Exception as e:
        st.error(f"Error reading {file_path}: {str(e)}")
        return ""

def load_pdfs_from_folder():
    # Get all PDF files from the data folder
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        st.error("No PDF files found in the data folder!")
        return
    
    # Read all PDF files and create document objects
    documents = []
    for file_path in pdf_files:
        text = read_pdf(file_path)
        if text:
            filename = os.path.basename(file_path)
            documents.append(Document(text=text, metadata={"filename": filename}))
    
    if not documents:
        st.error("Could not extract text from any PDFs in the data folder!")
        return
    
    # Dimension for Gemini embeddings
    d = 768  # Gemini embedding dimension
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    # Initialize Groq LLM and Gemini embedding model
    llm = Groq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001", 
        api_key=GOOGLE_API_KEY
    )
    
    # Configure global settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    st.session_state.index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
    st.session_state.pdfs_loaded = True
    
    return len(documents)

def get_bot_response(user_input):
    if st.session_state.index is None:
        return "PDFs have not been loaded yet! Please check the data folder."
    
    query_engine = st.session_state.index.as_query_engine(
        response_mode="compact"
    )
    response = query_engine.query(SYSTEM_PROMPT + user_input)
    return str(response)

def clear_chat_history():
    st.session_state.chat_history = []

# Load PDFs automatically when the app starts
if not st.session_state.pdfs_loaded:
    with st.spinner("Loading PDFs from the data folder..."):
        num_docs = load_pdfs_from_folder()
        if num_docs:
            st.session_state.pdfs_loaded = True

# Sidebar
with st.sidebar:
    st.header("🧹 Clear Chat")
    if st.button("Clear Chat History"):
        clear_chat_history()
        st.rerun()

# Main content area
# Use HTML and CSS to display the image and text inline
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="data:image/png;base64,{image_to_base64(image)}" alt="Neo4py Logo" width="50" style="vertical-align: middle;">
        <h1 style="margin: 0;">Neo4py QA Bot</h1>
    </div>
    <p>Welcome to the Neo4py QA Bot!</p>
    """,
    unsafe_allow_html=True
)

# Check if data folder exists
if not os.path.exists("data"):
    st.warning("The 'data' folder does not exist! Please create it and add your PDF files.")

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about neo4py...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = get_bot_response(user_input)
        message_placeholder.markdown(full_response)
    
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})

if not st.session_state.chat_history and not st.session_state.pdfs_loaded:
    st.info("PDFs are being loaded. Please wait...")