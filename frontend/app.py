"""
Streamlit Frontend for RAG Knowledge Assistant
Production-ready UI with authentication, document upload, and chat interface
"""
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict, List

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .source-box {
        background-color: #fff9e6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #ffa726;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []


# API Functions
def api_call(endpoint: str, method: str = "GET", data: dict = None, files: dict = None) -> Optional[dict]:
    """Make API call with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, files=files, data=data)
            else:
                headers["Content-Type"] = "application/json"
                response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            return None
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            error_detail = response.json().get("detail", "Unknown error")
            st.error(f"API Error: {error_detail}")
            return None
    
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None


def login(username: str, password: str) -> bool:
    """Login user"""
    response = api_call("/auth/login", "POST", {"username": username, "password": password})
    if response:
        st.session_state.token = response["access_token"]
        # Get user profile
        user_data = api_call("/auth/me", "GET")
        if user_data:
            st.session_state.user = user_data
            return True
    return False


def register(username: str, email: str, password: str) -> bool:
    """Register new user"""
    response = api_call("/auth/register", "POST", {
        "username": username,
        "email": email,
        "password": password
    })
    if response:
        st.session_state.token = response["access_token"]
        # Get user profile
        user_data = api_call("/auth/me", "GET")
        if user_data:
            st.session_state.user = user_data
            return True
    return False


def logout():
    """Logout user"""
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.chat_history = []
    st.session_state.documents = []


def upload_document(file) -> bool:
    """Upload document"""
    files = {"file": (file.name, file, file.type)}
    response = api_call("/upload/", "POST", files=files)
    if response:
        st.success(f"✅ Uploaded: {response['filename']} ({response['chunk_count']} chunks)")
        return True
    return False


def query_documents(query: str) -> Optional[dict]:
    """Query documents"""
    return api_call("/query/", "POST", {"query": query})


def get_documents() -> List[dict]:
    """Get user's documents"""
    response = api_call("/upload/documents", "GET")
    return response if response else []


def delete_document(doc_id: int) -> bool:
    """Delete document"""
    response = api_call(f"/upload/documents/{doc_id}", "DELETE")
    return response is not None


def get_history() -> List[dict]:
    """Get query history"""
    response = api_call("/history/", "GET")
    return response["items"] if response else []


# UI Components
def render_auth_page():
    """Render login/register page"""
    st.markdown('<div class="main-header">🤖 RAG Knowledge Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your AI-powered document Q&A system</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary"):
            if username and password:
                with st.spinner("Logging in..."):
                    if login(username, password):
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
            else:
                st.warning("Please enter username and password")
    
    with tab2:
        st.subheader("Create Account")
        new_username = st.text_input("Username", key="register_username")
        new_email = st.text_input("Email", key="register_email")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Register", type="primary"):
            if new_username and new_email and new_password and confirm_password:
                if new_password != confirm_password:
                    st.error("❌ Passwords don't match")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters")
                else:
                    with st.spinner("Creating account..."):
                        if register(new_username, new_email, new_password):
                            st.success("✅ Account created successfully!")
                            st.rerun()
            else:
                st.warning("Please fill all fields")
    
    # Info section
    st.markdown("---")
    st.markdown("### 🚀 Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📄 Document Upload**")
        st.markdown("Upload PDF, TXT, DOCX")
    with col2:
        st.markdown("**💬 Smart Q&A**")
        st.markdown("Ask questions, get answers")
    with col3:
        st.markdown("**📚 Source Citations**")
        st.markdown("See where answers come from")


def render_main_app():
    """Render main application"""
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user['username']}")
        st.markdown(f"📧 {st.session_state.user['email']}")
        
        if st.button("Logout", type="secondary"):
            logout()
            st.rerun()
        
        st.markdown("---")
        
        # Document upload section
        st.markdown("### 📤 Upload Documents")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "docx"],
            help="Upload PDF, TXT, or DOCX files"
        )
        
        if uploaded_file:
            if st.button("Upload", type="primary"):
                with st.spinner("Processing document..."):
                    if upload_document(uploaded_file):
                        st.session_state.documents = get_documents()
                        st.rerun()
        
        st.markdown("---")
        
        # Documents list
        st.markdown("### 📚 My Documents")
        if st.button("Refresh Documents"):
            st.session_state.documents = get_documents()
        
        if not st.session_state.documents:
            st.session_state.documents = get_documents()
        
        if st.session_state.documents:
            for doc in st.session_state.documents:
                with st.expander(f"📄 {doc['filename'][:30]}..."):
                    st.write(f"**Type:** {doc['file_type']}")
                    st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                    st.write(f"**Chunks:** {doc['chunk_count']}")
                    st.write(f"**Uploaded:** {doc['uploaded_at'][:10]}")
                    
                    if st.button(f"Delete", key=f"delete_{doc['id']}"):
                        if delete_document(doc['id']):
                            st.success("Document deleted")
                            st.session_state.documents = get_documents()
                            st.rerun()
        else:
            st.info("No documents uploaded yet")
    
    # Main area
    st.markdown('<div class="main-header">💬 Ask Questions</div>', unsafe_allow_html=True)
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            # User message
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑 You:</strong>
                <p>{message['query']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant message
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong>
                <p>{message['response']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sources
            if message.get('sources'):
                with st.expander("📚 View Sources"):
                    for idx, source in enumerate(message['sources'], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {idx}: {source['document_name']}</strong><br>
                            Similarity: {source['similarity_score']:.2%}<br>
                            <em>{source['content']}</em>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Query input
    st.markdown("---")
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="E.g., What is the main topic of my documents?",
            label_visibility="collapsed"
        )
    
    with col2:
        ask_button = st.button("Ask", type="primary", use_container_width=True)
    
    if ask_button and query:
        with st.spinner("🔍 Searching documents and generating answer..."):
            response = query_documents(query)
            
            if response:
                st.session_state.chat_history.append(response)
                st.rerun()
    
    # History section
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# Main app logic
def main():
    if st.session_state.token is None:
        render_auth_page()
    else:
        render_main_app()


if __name__ == "__main__":
    main()

