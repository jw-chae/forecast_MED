import streamlit as st
# Remove the old agent import
# from r1_smolagent_rag import primary_agent 
# Import the new LangChain RAG chain
from r1_smolagent_rag import rag_chain 

def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(prompt: str):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using the LangChain RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the LangChain RAG chain
                response = rag_chain.invoke(prompt) 
                st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

def display_sidebar():
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This Q&A bot uses a LangChain RAG (Retrieval Augmented Generation) 
        pipeline with DeepSeek to answer questions about your documents.
        
        The process:
        1. Your query is used to search through document chunks.
        2. Most relevant chunks are retrieved from the vector store.
        3. A reasoning model (DeepSeek) generates an answer based on the query and retrieved context, 
           acting as a medical assistant.
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

def main():
    # Set up Streamlit page
    st.set_page_config(page_title="Medical Doc Q&A (LangChain+DeepSeek)", layout="wide")
    st.title("Medical Document Q&A Bot")

    # Initialize chat history
    init_chat_history()

    # Display chat interface
    display_chat_history()
    display_sidebar()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        handle_user_input(prompt)

if __name__ == "__main__":
    main()
