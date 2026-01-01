"""
Streamlit web UI for FAQ Chatbot
Simple and interactive chat interface
"""

import streamlit as st
from chatbot import FAQChatbot, FAQ_DATABASE, TOPIC
import time

# Page configuration
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .similarity-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .high {
        background-color: #4CAF50;
        color: white;
    }
    .medium {
        background-color: #FF9800;
        color: white;
    }
    .low {
        background-color: #f44336;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = FAQChatbot()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🤖 FAQ Chatbot")
    st.caption(f"Supporting {TOPIC}")
with col2:
    if st.button("🔄 Clear Chat", help="Clear conversation history"):
        st.session_state.messages = []
        st.session_state.chatbot.clear_history()
        st.rerun()

# Sidebar
with st.sidebar:
    st.header("📊 Information")
    
    stats = st.session_state.chatbot.get_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("FAQs Available", stats['total_faqs'])
    with col2:
        st.metric("Queries Asked", stats['total_user_queries'])
    
    st.divider()
    
    st.subheader("🔍 Browse FAQs")
    if st.checkbox("Show all FAQs"):
        st.write("### Available Questions:")
        for i, faq in enumerate(st.session_state.chatbot.get_all_faqs(), 1):
            with st.expander(f"{i}. {faq['question']}", expanded=False):
                st.write(faq['answer'])
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Lower threshold = more results but less precise"
    )
    
    show_confidence = st.checkbox("Show confidence scores", value=True)
    show_similar = st.checkbox("Show similar questions", value=False)

# Main chat area
st.subheader("💬 Chat")

# Display chat history
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.markdown(
            f"""
            <div class="chat-message user-message">
                <div style="flex: 1;">
                    <strong>You:</strong><br/>
                    {message['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        confidence_badge = ""
        if show_confidence and 'match_info' in message:
            confidence = message['match_info']['confidence']
            score = message['match_info']['similarity_score']
            confidence_badge = f'<span class="similarity-badge {confidence}">Confidence: {confidence.upper()} ({score:.2%})</span>'
        
        matched_question = ""
        if 'match_info' in message and message['match_info']['matched_question']:
            matched_question = f"<br/><small><strong>Matched:</strong> {message['match_info']['matched_question']}</small>"
        
        st.markdown(
            f"""
            <div class="chat-message bot-message">
                <div style="flex: 1;">
                    <strong>Bot:</strong><br/>
                    {message['content']}
                    {confidence_badge}
                    {matched_question}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# User input
user_input = st.chat_input("Ask a question about our products and services...")

if user_input:
    # Add user message to display
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input
    })
    
    # Get bot response
    with st.spinner("Finding best match..."):
        response = st.session_state.chatbot.get_response(user_input, threshold=similarity_threshold)
    
    # Add bot response to display
    st.session_state.messages.append({
        'role': 'bot',
        'content': response['answer'],
        'match_info': {
            'similarity_score': response['similarity_score'],
            'matched_question': response['matched_question'],
            'confidence': response['confidence']
        }
    })
    
    # Show similar FAQs if enabled
    if show_similar:
        st.divider()
        with st.expander("📚 Similar Questions", expanded=True):
            similar = st.session_state.chatbot.get_similar_faqs(user_input, top_n=3)
            if similar:
                for i, match in enumerate(similar, 1):
                    score = match['similarity_score']
                    st.write(f"**{i}. {match['matched_question']}** (Similarity: {score:.2%})")
                    st.write(match['answer'])
                    st.divider()
            else:
                st.info("No similar questions found.")
    
    # Rerun to display new messages
    st.rerun()

# Footer
st.divider()
st.caption("FAQ Chatbot v1.0 | Using TF-IDF and Cosine Similarity for matching")
