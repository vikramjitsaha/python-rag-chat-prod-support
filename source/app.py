"""
Production Support RAG Chatbot - Streamlit Application
A polite and helpful AI assistant for production support queries
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add parent directory to path to import from source
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))

from rag_retriever import RAGRetriever
from data_loader import DataLoader
import time

# Change working directory to project root
os.chdir(project_root)


# Page configuration
st.set_page_config(
    page_title="Production Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_system():
    """
    Initialize the RAG system and check if data is loaded
    """
    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'system_ready' not in st.session_state:
        st.session_state.system_ready = False
    
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False
    
    # Check if data is already loaded (from previous session)
    if not st.session_state.system_ready and os.path.exists("Input.xlsx"):
        try:
            # Try to initialize retriever if database exists
            retriever = RAGRetriever()
            if retriever.collection.count() > 0:
                st.session_state.retriever = retriever
                st.session_state.system_ready = True
                st.session_state.data_loaded = True
                st.session_state.file_uploaded = True
        except:
            pass


def save_uploaded_file(uploaded_file):
    """
    Save the uploaded file to disk
    """
    try:
        with open("Input.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False


def load_data_to_vectordb():
    """
    Load Excel data into vector database
    """
    try:
        with st.spinner("Loading Excel data into vector database... This may take a few minutes."):
            loader = DataLoader()
            num_docs = loader.load_data_to_vectordb()
            
        st.success(f"✓ Successfully loaded {num_docs} documents into the vector database!")
        
        # Reinitialize the retriever
        st.session_state.retriever = RAGRetriever()
        st.session_state.system_ready = True
        st.session_state.data_loaded = True
        
        return num_docs
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def display_sidebar():
    """
    Display sidebar with system information and controls
    """
    with st.sidebar:
        st.title("🤖 Production Support Bot")
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        
        # API Key status
        if st.session_state.get('api_key_configured', False):
            st.success("✓ API Key Configured")
        else:
            st.error("❌ API Key Required")
        
        # File upload status
        if st.session_state.get('file_uploaded', False):
            st.success("✓ File Uploaded")
        else:
            st.info("📁 No file uploaded")
        
        # Data loading status
        if st.session_state.get('data_loaded', False):
            st.success("✓ Data Loaded to Vector DB")
        else:
            st.info("📊 Data not loaded yet")
        
        # System ready status
        if st.session_state.get('system_ready', False):
            st.success("✓ System Ready for Chat")
            
            # Get collection info
            try:
                collection = st.session_state.retriever.collection
                count = collection.count()
                st.info(f"� Knowledge Base: {count} documents")
            except:
                pass
        else:
            st.warning("⚠ System Not Ready")
        
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        
        st.session_state.top_k = st.slider(
            "Number of context documents",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of relevant documents to retrieve for each query"
        )
        
        st.session_state.show_context = st.checkbox(
            "Show retrieved context",
            value=False,
            help="Display the source documents used to generate the answer"
        )
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # API Key configuration (MANDATORY)
        st.subheader("🔑 Google Gemini API Key")
        st.caption("⚠️ Required for AI-powered responses")
        
        api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Required: Enter your Google Gemini API key to use this application",
            placeholder="AIza..."
        )
        
        if api_key and len(api_key) > 10:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.session_state.api_key_configured = True
            st.success("✓ API Key configured")
        else:
            st.session_state.api_key_configured = False
            if api_key:
                st.error("❌ Invalid API key format")
            else:
                st.warning("⚠️ Please enter your API key to continue")
        
        if not st.session_state.get('api_key_configured', False):
            st.info("💡 Get your free API key from: https://makersuite.google.com/app/apikey")
        
        st.markdown("---")
        st.caption("Production Support RAG System v2.0")


def display_chat_interface():
    """
    Display the main chat interface
    """
    st.title("💬 Production Support Assistant")
    st.markdown("Ask me anything about production support. I'll answer based on the knowledge base.")
    
    # Step 0: Check API Key (MANDATORY)
    if not st.session_state.get('api_key_configured', False):
        st.warning("⚠️ **Please configure your Google Gemini API Key in the sidebar to continue.**")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            ### Getting Started
            
            1. **Get your API Key** 
               - Visit: https://makersuite.google.com/app/apikey
               - Sign in with your Google account
               - Click "Create API Key"
               - Copy the API key
            
            2. **Enter the API Key**
               - Look at the sidebar on the left ⬅️
               - Find "🔑 Google Gemini API Key" section
               - Paste your API key
            
            3. **Start chatting!**
               - Upload your Excel file
               - Load data to vector database
               - Ask questions
            """)
        
        return
    
    # Step 1: File Upload
    if not st.session_state.get('file_uploaded', False):
        st.info("👋 Welcome! Please upload your Input.xlsx file to get started.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader(
                "Upload Input.xlsx",
                type=['xlsx'],
                help="Upload your Excel file containing production support data"
            )
            
            if uploaded_file is not None:
                if st.button("📤 Upload File", use_container_width=True, type="primary"):
                    if save_uploaded_file(uploaded_file):
                        st.session_state.file_uploaded = True
                        st.success("✓ File uploaded successfully!")
                        time.sleep(1)
                        st.rerun()
        
        return
    
    # Step 2: Load Data to Vector Database
    if not st.session_state.get('data_loaded', False):
        st.info("📊 File uploaded! Now load the data into the vector database.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Ready to Load Data")
            st.write(f"**File:** Input.xlsx")
            st.write("This process will read your Excel file and create a searchable knowledge base.")
            
            if st.button("🚀 Load Data to Vector Database", use_container_width=True, type="primary"):
                num_docs = load_data_to_vectordb()
                if num_docs:
                    time.sleep(2)
                    st.rerun()
        
        # Option to re-upload file
        st.markdown("---")
        if st.button("🔄 Upload a Different File"):
            st.session_state.file_uploaded = False
            if os.path.exists("Input.xlsx"):
                os.remove("Input.xlsx")
            st.rerun()
        
        return
    
    # Step 3: Chat Interface (System Ready)
    if not st.session_state.get('system_ready', False):
        st.warning("⚠ System is loading... Please wait.")
        return
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message['content'])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message['content'])
                    
                    # Show context if enabled
                    if st.session_state.get('show_context', False) and 'context' in message:
                        with st.expander("📚 View Source Documents"):
                            for j, doc in enumerate(message['context']):
                                st.markdown(f"**Document {j+1}:**")
                                st.text(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])
                                if doc.get('distance'):
                                    st.caption(f"Relevance Score: {1 - doc['distance']:.2f}")
                                st.markdown("---")
    
    # Chat input
    user_input = st.chat_input("Type your question here...", key="user_input")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Get response from RAG system
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    response, context = st.session_state.retriever.answer_query(
                        user_input,
                        top_k=st.session_state.get('top_k', 3)
                    )
                    
                    st.markdown(response)
                    
                    # Add assistant message to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'context': context
                    })
                    
                    # Show context if enabled
                    if st.session_state.get('show_context', False) and context:
                        with st.expander("📚 View Source Documents"):
                            for j, doc in enumerate(context):
                                st.markdown(f"**Document {j+1}:**")
                                st.text(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])
                                if doc.get('distance'):
                                    st.caption(f"Relevance Score: {1 - doc['distance']:.2f}")
                                st.markdown("---")
                    
                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': error_msg
                    })
        
        st.rerun()
    
    # Option to reload data
    with st.expander("🔧 Advanced Options"):
        st.write("**Reload Data:** Upload a new file and reload the vector database")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Upload New File", use_container_width=True):
                st.session_state.file_uploaded = False
                st.session_state.data_loaded = False
                st.session_state.system_ready = False
                st.session_state.chat_history = []
                if os.path.exists("Input.xlsx"):
                    os.remove("Input.xlsx")
                st.rerun()
        with col2:
            if st.button("♻️ Reload Current File", use_container_width=True):
                st.session_state.data_loaded = False
                st.session_state.system_ready = False
                st.session_state.chat_history = []
                st.rerun()


def main():
    """
    Main application entry point
    """
    # Initialize system
    initialize_system()
    
    # Display sidebar
    display_sidebar()
    
    # Display chat interface
    display_chat_interface()


if __name__ == "__main__":
    main()
