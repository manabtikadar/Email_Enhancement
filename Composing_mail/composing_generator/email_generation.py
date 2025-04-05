from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
load_dotenv()



class EmailOutput(BaseModel):
    From: str = Field(description="Sender's email address")
    To: str = Field(description="Recipient's email address")
    Subject: str = Field(description="Email subject")
    Body: str = Field(description="Generated email content")

template = """
You are an expert **{email_type} Email Assistant**. Your task is to generate a well-structured and professional email body based on the provided details.

### **Email Type:** {email_type}

### **Email Details (Extract Required Fields)**
{email}

### **User Query**
{query}

### **Retrieved Context**
{context}

---

## **Task:**
- Compose a **natural and professional email body** relevant to the subject and query.
- Ensure **clarity, conciseness, and accuracy**.
- Adapt tone and content **based on the specified email type**.
- **Extract key details (From, To, Subject) from the given email details.**

---

## **Rules:**
    - Maintain a professional tone suitable for the given email type.
    - Avoid predefined templates; generate a fresh response based on context.
    - Ensure the response is well-structured and relevant.
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["email_type", "email", "query","context"]
)

structured_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", 
                                        temperature=0, 
                                        google_api_key=os.getenv("GOOGLE_API_KEY_1")).bind_tools(
                                            tools=[EmailOutput]
                                     )


email_generation_chain = (
    prompt
    | structured_llm
)

