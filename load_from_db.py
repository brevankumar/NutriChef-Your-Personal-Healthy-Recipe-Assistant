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
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.node_parser import SentenceWindowNodeParser, SimpleNodeParser
from sentence_transformers import SentenceTransformer
from llama_index.memory import ChatMemoryBuffer
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent



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
        {"role": "assistant", "content": "Ask me a question to get healthy recipes; Example for a query like {Provide recipe for spinach with recipe name, ingredients, preparation and nutrition facts}"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the data – hang tight! This should take 1-2 minutes."):
        
        # load from disk
        db2 = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db2.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        

        # base Query Engine LLM
        llm = Gemini(api_key=os.getenv("google_api_key"),model='gemini-pro')

        # fine-tuned Embeddings model
        embed_model = HuggingFaceEmbedding(model_name='Revankumar/fine_tuned_embeddings_for_healthy_recipes')

        # fine-tuned ServiceContext
        ctx = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
        )
        

        index = VectorStoreIndex.from_vector_store(
            vector_store,
            service_context=ctx,
        )
        
        return index

index = load_data()

query_engine = index.as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="recipe",
            description=(
                "Provides information about healthy recipes."
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    )
]


# Add Context
context = """\
  You are a healthy recipe assistant who is an expert on the providing users with nutritious recipes.\
     You will answer questions about number of servings, ingredients, preparation instructions, and nutrition facts. \
 """

llm = Gemini(api_key=os.getenv("google_api_key"),model='gemini-pro')

agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    context=context
)


if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history



