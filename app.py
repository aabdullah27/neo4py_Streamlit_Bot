import streamlit as st
from PIL import Image
import base64
from io import BytesIO
import faiss
import os
import glob
from dotenv import load_dotenv
import pymupdf4llm
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini
from functools import lru_cache

# Load environment variables once
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Set constants
EMBEDDING_DIMENSION = 768
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "models/gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"

# System prompt
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

# Helper function to convert image to base64 (cached to avoid repeated conversions)
@lru_cache(maxsize=1)
def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

# Initialize Streamlit session state
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "index" not in st.session_state:
        st.session_state.index = None
    if "pdfs_loaded" not in st.session_state:
        st.session_state.pdfs_loaded = False

# Read PDF with error handling
def read_pdf(file_path):
    try:
        return pymupdf4llm.to_markdown(file_path)
    except Exception as e:
        st.error(f"Error reading {file_path}: {str(e)}")
        return ""

# Load PDFs and create vector index
def load_pdfs_from_folder():
    # Get all PDF files from the data folder
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        st.error("No PDF files found in the data folder!")
        return 0
    
    # Read all PDF files and create document objects
    documents = []
    for file_path in pdf_files:
        text = read_pdf(file_path)
        if text:
            documents.append(Document(text=text, metadata={"filename": os.path.basename(file_path)}))
    
    if not documents:
        st.error("Could not extract text from any PDFs in the data folder!")
        return 0
    
    # Create vector store
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    
    # Initialize embedding model
    embed_model = GeminiEmbedding(model_name=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
    
    # Configure global settings
    Settings.embed_model = embed_model
    
    # Initialize default LLM - try Groq first
    try:
        Settings.llm = Groq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
    except Exception as e:
        st.warning(f"Failed to initialize Groq: {str(e)}. Falling back to Gemini.")
        try:
            Settings.llm = Gemini(model=GEMINI_MODEL, api_key=GOOGLE_API_KEY)
        except Exception as e2:
            st.error(f"Failed to initialize any LLM. Please check your API keys. Error: {str(e2)}")
            return 0
    
    # Create and store index
    st.session_state.index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
    st.session_state.pdfs_loaded = True
    
    return len(documents)

# Get response with Groq primary and Gemini fallback
def get_bot_response(user_input):
    if st.session_state.index is None:
        return "PDFs have not been loaded yet! Please check the data folder."
    
    # Build context from recent messages (last 5 interactions)
    context_str = ""
    if st.session_state.chat_history:
        recent = st.session_state.chat_history[-10:]  # 5 interactions = 10 messages
        for i in range(0, len(recent), 2):
            if i+1 < len(recent):
                context_str += f"### Previous Interaction:\n**User**: {recent[i]['content']}\n**Assistant**: {recent[i+1]['content']}\n\n"
    
    # Combine system prompt, context, and current question
    full_query = f"{SYSTEM_PROMPT}\n\n{context_str}\n### New Question:\n{user_input}"
    
    # Try Groq first, fall back to Gemini
    try:
        Settings.llm = Groq(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        query_engine = st.session_state.index.as_query_engine(response_mode="compact")
        return str(query_engine.query(full_query))
    except Exception as e:
        st.warning(f"Groq query failed: {str(e)}. Falling back to Gemini.")
        try:
            Settings.llm = Gemini(model=GEMINI_MODEL, api_key=GOOGLE_API_KEY)
            query_engine = st.session_state.index.as_query_engine(response_mode="compact")
            return str(query_engine.query(full_query))
        except Exception as e2:
            return f"Both Groq and Gemini failed. Error: {str(e2)}"

# Clear chat history
def clear_chat_history():
    st.session_state.chat_history = []

# Main function
def main():
    # Set page config as the first Streamlit command
    st.set_page_config(
        page_title="Neo4py QA Bot", 
        page_icon="🤖", 
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar with only clear chat option
    with st.sidebar:
        st.header("🧹 Clear Chat")
        if st.button("Clear Chat History"):
            clear_chat_history()
            st.rerun()
    
    # Load image only once and display header
    image_path = "assets/minion.png"
    if os.path.exists(image_path):
        base64_image = image_to_base64(image_path)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px;">
                <img src="data:image/png;base64,{base64_image}" alt="Neo4py Logo" width="50" style="vertical-align: middle;">
                <h1 style="margin: 0;">Neo4py QA Bot</h1>
            </div>
            <p>Welcome to the Neo4py QA Bot!</p>
            """,
            unsafe_allow_html=True
        )
    else:
        st.title("Neo4py QA Bot")
        st.write("Welcome to the Neo4py QA Bot!")
    
    # Check data folder and load PDFs
    if not os.path.exists("data"):
        st.warning("The 'data' folder does not exist! Please create it and add your PDF files.")
    elif not st.session_state.pdfs_loaded:
        with st.spinner("Loading PDFs from the data folder..."):
            num_docs = load_pdfs_from_folder()
            if num_docs:
                st.success(f"Loaded {num_docs} documents successfully!")
    
    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    user_input = st.chat_input("Ask a question about neo4py...")
    if user_input:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                full_response = get_bot_response(user_input)
                st.markdown(full_response)
        
        # Add response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()