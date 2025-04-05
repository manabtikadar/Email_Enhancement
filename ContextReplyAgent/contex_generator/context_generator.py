from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

prompt = PromptTemplate(
    template="""
    You are an expert email assistant. Your job is to generate a professional, clear, and context-aware email **reply** based on the user's current query and prior email conversation and retrieved context data from documents.

    ---

    **Previous Email Thread:**
    {previous_response}

    **Current Incoming Email:**
    {email_input}

    **User Query / Intention:**
    {query}

    **Retrieved Context:**
    {context}
    ---

    ### Task:
    - Understand the user's intent from the query.
    - Use the context of previous conversation if provided.
    - Maintain formal tone, professionalism, and clarity.
    - Keep the subject line consistent unless instructed otherwise.
    - Output strictly in JSON format.

    ### Format (Strict JSON only):
    ```json
    {{
    "From": "<original sender in email_input>",
    "To": "<recipient in email_input>",
    "Subject": "<same or refined subject from email_input>",
    "Body": "<LLM-generated professional reply>"
    }}
    """,
    input_variables=["previous_response","email_input","query","context"]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key = os.getenv("GOOGLE_API_KEY"),
    temperature = 0
)

context_reply_chain = (
    prompt
    | llm
    | JsonOutputParser()
)