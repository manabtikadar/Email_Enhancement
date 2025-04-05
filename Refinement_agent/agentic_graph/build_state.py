from typing_extensions import TypedDict
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from Refinement_generator import email_refiner_chain
from grader import refinementGrader_chain
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__),"..")))

class AgentState(TypedDict):
    """Agent state representation."""
    email: dict[str,str]
    query: str
    needingImprovement: str
    ImprovementList: list[str]
    refined_email: dict[str, str]

def refinement_grader(state:AgentState)->AgentState:
    """Grader function to evaluate the email and query."""
    email = state["email"]
    query = state["query"]

    response = refinementGrader_chain.invoke({
        "query":query,
        "email_input":email
    })

    result = json.loads(response.additional_kwargs["tool_calls"][0]["function"]["arguments"])
    needing_improvement = result["needingImprovement"]
    improvement_list = result["ImprovementList"]

    return {
        **state,
        "needingImprovement": needing_improvement,
        "ImprovementList": improvement_list
    }

def user_friendly_generator(state:AgentState)->AgentState:
    """Generator function to refine the email."""
    email = state["email"]
    query = state["query"]
    improvement_list = state["ImprovementList"]

    response = email_refiner_chain.invoke({
        "email":email,
        "ImprovementList":improvement_list
    })

    return {
        **state,
        "refined_email":response
    }


