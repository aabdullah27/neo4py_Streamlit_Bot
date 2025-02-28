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
from llama_index.llms.gemini import Gemini  # Changed from Groq to Gemini
import pymupdf4llm
import os
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index" not in st.session_state:
    st.session_state.index = None
if "pdfs_loaded" not in st.session_state:
    st.session_state.pdfs_loaded = False

# Updated system prompt with memory instructions
SYSTEM_PROMPT = """
You are a highly knowledgeable and helpful assistant specialized in Neo4py and related topics. You provide clear, concise, and detailed answers about Neo4py, focusing on accuracy and professionalism.

Casual Interaction: Respond naturally to casual greetings and small talk (e.g., "Hello," "How are you?") without diving into technical details about Neo4py unless specifically asked.

Technical Queries: When the user asks about Neo4py, analyze the prompt thoroughly and respond with in-depth, accurate, and structured information in Markdown format. Your answers should be:

- Detailed: Provide explanations, code snippets, and best practices where relevant.
- Concise and Clear: Avoid unnecessary information while covering the topic comprehensively.
- Helpful: Offer step-by-step guidance if needed, focusing exclusively on Neo4py and avoiding unrelated topics.
- Unknown Information: If you do not know the answer, admit it honestly and avoid speculation.

Tone: Maintain a professional yet approachable tone, ensuring that responses are tailored to the context of the user's queries.
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
    
    # Initialize Gemini LLM and embedding model
    llm = Gemini(model="models/gemini-2.0-flash", api_key=GOOGLE_API_KEY)  # Changed from Groq to Gemini
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
    
    # Get last 5 interactions (10 messages)
    chat_history = st.session_state.chat_history
    recent_messages = chat_history[-10:]  # Each interaction has 2 messages
    
    # Build context from recent messages
    context_str = ""
    for i in range(0, len(recent_messages), 2):
        if i+1 < len(recent_messages):
            user_msg = recent_messages[i]["content"]
            assistant_msg = recent_messages[i+1]["content"]
            context_str += f"### Previous Interaction:\n**User**: {user_msg}\n**Assistant**: {assistant_msg}\n\n"
    
    # Combine system prompt, context, and current question
    full_query = f"{SYSTEM_PROMPT}\n\n{context_str}\n### New Question:\n{user_input}"
    
    query_engine = st.session_state.index.as_query_engine(
        response_mode="compact"
    )
    response = query_engine.query(full_query)
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