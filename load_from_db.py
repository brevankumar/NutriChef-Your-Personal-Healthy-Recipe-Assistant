import os
from dotenv import load_dotenv
from llama_index import download_loader
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from IPython.display import Markdown, display
from llama_index.llms import Gemini
import chromadb
import streamlit as st


# Enable Logging
import logging
import sys

#You can set the logging level to DEBUG for more verbose output,
# or use level=logging.INFO for less detailed information.
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Load environment variables from the .env file
load_dotenv()


st.header("NutriChef: Your Personal Healthy Recipe Assistant")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question to get healthy recipes"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the data – hang tight! This should take 1-2 minutes."):
        # load from disk
        db2 = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db2.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        llm = Gemini(api_key='AIzaSyA903hLQGcsRu0IrCVKJeoxV8JPwMzPCXk',model='gemini-pro')
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

        service_context = ServiceContext.from_defaults(embed_model=embed_model,llm=llm)

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            service_context=service_context,
        )
        
        return index

index = load_data()


chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)


if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history




















# Query Data
query_engine = index.as_query_engine()