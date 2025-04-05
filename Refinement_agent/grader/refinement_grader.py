from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

class RefinementGrader(BaseModel):
    needingImprovement: str = Field(
        ..., description="Return 'yes' if the user's query is asking for refinement or correction of the email; otherwise, return 'no'."
    )
    ImprovementList: list[str] = Field(
        ..., description="List specific areas that need improvement based on the user's email input. Possible areas include: "
                         "- Tone Improvement, "
                         "- Grammar Improvement, "
                         "- Clarity Improvement, "
                         "- Too vague, "
                         "- Lacks professionalism, "
                         "- Missing subject line, "
                         "- No introduction or closing, "
                         "- Lacks structure."
    )


refinement_prompt = PromptTemplate(
    template="""
        You are an expert in email refinement and communication improvement.

        Analyze the following:
        1. The user's email content
        2. The user's query

        Determine whether the user is asking for refinement or correction of the email.

        If refinement is required, provide:
        - **needingImprovement**: "yes" if the query indicates a request to refine or correct the email; otherwise, "no".
        - **ImprovementList**: A list of specific areas needing improvement based on the email content.

        User Email:
        {email_input}

        User Query:
        {query}
    """,
    input_variables=["query", "email_input"]
)


structured_llm = ChatCohere(
    temperature=0,
    cohere_api_key=os.getenv("SUBAGENT_COHERE_API_KEY")
).bind_tools(
    tools=[RefinementGrader]
)

refinementGrader_chain = (
    refinement_prompt 
    | structured_llm
)