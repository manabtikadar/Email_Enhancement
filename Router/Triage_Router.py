from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal
from dotenv import load_dotenv
import os

load_dotenv()

class TriageRouter(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning behind why the email is categorized as Ignore, Respond, or Notify."
    )
    classification: Literal["Ignore", "Respond", "Notify"] = Field(
        description="""
        The classification of the email:
        - 'Ignore': Irrelevant, spam, contains sensitive personal data (e.g., phone number, OTP, passwords), or requires no action.
        - 'Notify': Important information that does not require a response.
        - 'Respond': The email needs a direct reply or action from the recipient.
        """
    )

triage_router_prompt = PromptTemplate(
    template="""
You are an expert email triage assistant. Your task is to classify the email into one of the following:

1. **Ignore** → 
   - Irrelevant, spam, or requires no action.
   - OR contains personal sensitive information such as phone numbers, OTPs, passwords, etc.

2. **Notify** → 
   - Contains important information, updates, or announcements that the user should be aware of, 
     but does not require a response.

3. **Respond** → 
   - Clearly requests information, requires action, or expects a reply from the recipient.

---

**Message Query**:
{messages_query}

---

### Task:
Analyze the content carefully and determine the appropriate classification.

**Response Format (strict)**:
Reasoning: <step-by-step explanation>
Classification: <Ignore / Notify / Respond>
""",
    input_variables=["messages_query"]
)

llm = ChatCohere(
    temperature=0.0,
    cohere_api_key=os.getenv("SUBAGENT_COHERE_API_KEY"),
).bind_tools(
    tools = [TriageRouter]
)

triage_router_chain = (
    triage_router_prompt 
    | llm
)