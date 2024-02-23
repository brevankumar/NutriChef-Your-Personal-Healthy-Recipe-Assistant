import os
from dotenv import load_dotenv
from llama_index.legacy import VectorStoreIndex, ServiceContext
from llama_index.legacy.vector_stores import ChromaVectorStore
from llama_index.legacy.storage.storage_context import StorageContext
from IPython.display import Markdown, display
from llama_index.legacy.llms import Gemini
import chromadb
import streamlit as st

from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.legacy.prompts import PromptTemplate
from llama_index.legacy import ServiceContext




load_dotenv() ## load all the environment variables

import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


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
        llm = Gemini(api_key=os.getenv("GOOGLE_API_KEY"),model='gemini-pro')

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

query_engine = index.as_query_engine(response_mode="tree_summarize")


context_str = """Make most of your meal vegetables and fruits – ½ of your plate.
                Aim for color and variety, and remember that potatoes don’t count as vegetables on the Healthy Eating Plate because of their negative impact on blood sugar.

                Go for whole grains – ¼ of your plate.
                Whole and intact grains—whole wheat, barley, wheat berries, quinoa, oats, brown rice, and foods made with them, such as whole wheat pasta—have a milder effect on blood sugar and insulin than white bread, white rice, and other refined grains.

                Protein power – ¼ of your plate.
                Fish, poultry, beans, and nuts are all healthy, versatile protein sources—they can be mixed into salads, and pair well with vegetables on a plate. Limit red meat, and avoid processed meats such as bacon and sausage.

                Healthy plant oils – in moderation.
                Choose healthy vegetable oils like olive, canola, soy, corn, sunflower, peanut, and others, and avoid partially hydrogenated oils, which contain unhealthy trans fats. Remember that low-fat does not mean “healthy.”

                Drink water, coffee, or tea.
                Skip sugary drinks, limit milk and dairy products to one to two servings per day, and limit juice to a small glass per day."""


new_summary_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query in the style of a nutritionist.\n"
    "Query: {query_str}\n"
    "Answer: "
)
new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)

query_engine.update_prompts(
    {"response_synthesizer:summary_template": new_summary_tmpl}
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
            response = query_engine.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history


