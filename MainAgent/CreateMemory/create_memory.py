from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnableConfig
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
store = InMemoryVectorStore(
    embedding=embedding_model
)


def get_user_id(config: RunnableConfig) ->str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id

