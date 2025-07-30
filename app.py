"""
Streamlit frontend for Sapphire RAG chatbot.
Simple and clean interface for bot and user communication.
"""

import streamlit as st
import requests
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="Sapphire Fashion Assistant",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }
    
    .main-header {
        text-align: center;
        color: #1a1a1a;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center;
        color: #4a4a4a;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-weight: 500;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-weight: 500;
    }
    
    .message-time {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.5rem;
        text-align: right;
    }
    
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #E0E0E0;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        color: #333333 !important;
        font-weight: 500;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #999999 !important;
        font-weight: 400;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Quick action buttons */
    .quick-action-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .quick-action-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }
    
    .error-message {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .info-message {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .loading-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #1565c0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        font-weight: 500;
        text-align: center;
    }
    
    .loading-dots::after {
        content: '';
        animation: dots 1.5s steps(5, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% {
            color: rgba(0,0,0,0);
            text-shadow:
                .25em 0 0 rgba(0,0,0,0),
                .5em 0 0 rgba(0,0,0,0);
        }
        40% {
            color: #1565c0;
            text-shadow:
                .25em 0 0 rgba(0,0,0,0),
                .5em 0 0 rgba(0,0,0,0);
        }
        60% {
            text-shadow:
                .25em 0 0 #1565c0,
                .5em 0 0 rgba(0,0,0,0);
        }
        80%, 100% {
            text-shadow:
                .25em 0 0 #1565c0,
                .5em 0 0 #1565c0;
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_connected" not in st.session_state:
    st.session_state.api_connected = False

if "bot_initializing" not in st.session_state:
    st.session_state.bot_initializing = True

def check_api_connection():
    """Check if the API is running and accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.api_connected = True
            st.session_state.bot_initializing = False
            return True
    except requests.exceptions.RequestException:
        pass
    
    st.session_state.api_connected = False
    return False

def send_message(message: str, include_sources: bool = False):
    """Send a message to the chatbot API."""
    try:
        payload = {
            "message": message,
            "session_id": st.session_state.session_id,
            "include_sources": include_sources
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def display_message(message: Dict[str, Any], is_user: bool = True):
    """Display a chat message."""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong> {message['content']}
            <div class="message-time">{message['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_info = ""
        if message.get('sources'):
            sources_info = f"<br><small><em>Sources: {len(message['sources'])} references used</em></small>"
        
        st.markdown(f"""
        <div class="bot-message">
            <strong>Sapphire Assistant:</strong> {message['content']}{sources_info}
            <div class="message-time">{message['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)

def clear_conversation():
    """Clear the current conversation."""
    try:
        response = requests.delete(f"{API_BASE_URL}/conversations/{st.session_state.session_id}")
        if response.status_code == 200:
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.success("Conversation cleared!")
        else:
            st.error("Failed to clear conversation")
    except requests.exceptions.RequestException as e:
        st.error(f"Error clearing conversation: {str(e)}")

def main():
    """Main Streamlit application."""
    
    # Initialize user name in session state
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    
    if "greeting_sent" not in st.session_state:
        st.session_state.greeting_sent = False
    
    # Header
    st.markdown('<h1 class="main-header">💎 Sapphire Fashion Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your personal fashion consultant for Sapphire\'s latest collections</p>', unsafe_allow_html=True)
    
    # Check API connection
    api_status = check_api_connection()
    
    if not api_status:
        st.markdown("""
        <div class="error-message">
            <strong>⚠️ API Connection Error</strong><br>
            The chatbot API is not running. Please start the FastAPI server first:<br>
            <code>python main.py</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Quick Start Guide")
        st.markdown("""
        1. **Start the API server:**
           ```bash
           cd sapphire_chatbot
           python main.py
           ```
        
        2. **Set your Groq API key:**
           ```bash
           export GROQ_API_KEY="your-groq-api-key-here"
           ```
        
        3. **Refresh this page** once the API is running
        """)
        return
    
    # Show loading message only during initial bot setup
    if st.session_state.bot_initializing:
        st.markdown("""
        <div class="loading-message">
            <strong>🤖 Sapphire Assistant is initializing<span class="loading-dots">...</span></strong><br>
            <small>Setting up your personalized shopping experience</small>
        </div>
        """, unsafe_allow_html=True)
    
    # User name input (only show if not set)
    if not st.session_state.user_name:
        st.markdown("### Welcome to Sapphire!")
        user_name = st.text_input("Please enter your name to get started:", placeholder="Enter your name here")
        
        if st.button("Start Chat", type="primary") and user_name.strip():
            st.session_state.user_name = user_name.strip()
            st.session_state.greeting_sent = False
            st.rerun()
        
        if user_name.strip() == "":
            st.info("Please enter your name to begin chatting with our assistant.")
            return
    
    # Send initial greeting if not sent yet
    if st.session_state.user_name and not st.session_state.greeting_sent:
        greeting_message = {
            "content": f"Hi {st.session_state.user_name}! Welcome to Sapphire Online. How can we assist you today?",
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "sources": []
        }
        st.session_state.messages.append(greeting_message)
        st.session_state.greeting_sent = True
    
    # Sidebar with options
    with st.sidebar:
        st.header("Chat Options")
        
        # User info
        st.markdown(f"**Welcome, {st.session_state.user_name}!** 👋")
        
        include_sources = st.checkbox(
            "Show sources", 
            value=False,
            help="Include source information in responses"
        )
        
        st.divider()
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            clear_conversation()
        
        if st.button("👤 Change Name", use_container_width=True):
            st.session_state.user_name = ""
            st.session_state.greeting_sent = False
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        st.subheader("About")
        st.markdown("""
        This chatbot can help you with:
        - Product information
        - Pricing and availability
        - Sizing and fabric details
        - Return and exchange policies
        - Shipping information
        - General brand inquiries
        """)
        
        st.divider()
        
        # Session info
        st.subheader("Session Info")
        st.text(f"Session ID: {st.session_state.session_id[:8]}...")
        st.text(f"Messages: {len(st.session_state.messages)}")
    
    # Quick action buttons (show after greeting)
    if st.session_state.greeting_sent and len(st.session_state.messages) == 1:
        st.markdown("### Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📦 Check Order Status", use_container_width=True):
                st.session_state.quick_action_message = "I would like to check my order status"
                
            if st.button("🔄 Return & Exchange Policy", use_container_width=True):
                st.session_state.quick_action_message = "What is your return and exchange policy?"
        
        with col2:
            if st.button("📝 Register Case/Complaint", use_container_width=True):
                st.session_state.quick_action_message = "I need to register a case or complaint"
                
            if st.button("🔍 Check Case/Complaint Status", use_container_width=True):
                st.session_state.quick_action_message = "I want to check the status of my case or complaint"
        
        with col3:
            if st.button("📞 Contact Customer Support", use_container_width=True):
                st.session_state.quick_action_message = "How can I contact customer support?"
                
            if st.button("❓ Others", use_container_width=True):
                st.session_state.quick_action_message = "I have other questions"
        
        # Process quick action if selected
        if "quick_action_message" in st.session_state:
            message = st.session_state.quick_action_message
            del st.session_state.quick_action_message
            
            # Add user message
            user_message = {
                "content": message,
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "is_user": True
            }
            st.session_state.messages.append(user_message)
            
            # Get bot response
            with st.spinner("Getting response..."):
                response = send_message(message, include_sources)
                
                if response:
                    bot_message = {
                        "content": response["response"],
                        "timestamp": datetime.now().strftime("%I:%M %p"),
                        "sources": response.get("sources", [])
                    }
                    st.session_state.messages.append(bot_message)
            
            st.rerun()
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        if message.get("is_user", False):
            display_message(message, is_user=True)
        else:
            display_message(message, is_user=False)
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask me anything about Sapphire...",
                placeholder="e.g., What's your return policy? or Show me summer dresses",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("Send", use_container_width=True)
    
    # Process user input
    if send_button and user_input.strip():
        # Add user message to chat history
        user_message = {
            "content": user_input,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "is_user": True
        }
        st.session_state.messages.append(user_message)
        
        # Get bot response
        with st.spinner("Sapphire Assistant is typing..."):
            response = send_message(user_input, include_sources)
        
        if response:
            # Add bot response to chat history
            bot_message = {
                "content": response["response"],
                "sources": response.get("sources", []),
                "timestamp": datetime.now().strftime("%I:%M %p")
            }
            st.session_state.messages.append(bot_message)
        
        # Rerun to show new messages
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Powered by Sapphire RAG Pipeline | Built with Streamlit & FastAPI</p>
        <p>For the best experience, ensure your Groq API key is configured</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()