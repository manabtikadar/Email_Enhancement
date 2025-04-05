from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

class EmailClassifier(BaseModel):
  email_type : Literal[
     "Technical",
      "Formal",
      "Informal",
      "Marketing",
      "Customer Support",
      "HR & Recruitment",
      "Finance & Billing",
      "Legal",
      "Personal"
  ] = Field(
    description="The type of email to classify the content into."
  )

prompt = PromptTemplate(
  template="""
You are an expert email classification assistant. Classify the following email into one of these categories:

- Technical: Technology, troubleshooting, software, or engineering topics.
- Formal: Official or structured professional messages.
- Informal: Casual, friendly, or conversational emails.
- Marketing: Promotions, advertisements, or newsletters.
- Customer Support: Help desk, ticket, or service issues.
- HR & Recruitment: Job applications, HR processes, interviews.
- Finance & Billing: Invoices, payments, budget-related communication.
- Legal: Policies, contracts, compliance, or legal concerns.
- Personal: Private, non-work-related communication.

Email:
{query}

Return only the category name (e.g., "Technical", "Personal", etc.) as structured JSON output.
""",
  input_variables=["query"],
)
  
llm = ChatCohere(
  temperature=0,
  cohere_api_key=os.getenv("CLASSIFIER_COHERE_API_KEY")
).bind_tools(
  tools=[EmailClassifier]
)


email_type_router_chain = (
  prompt
  | llm
)



