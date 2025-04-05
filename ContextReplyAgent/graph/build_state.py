import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from contex_generator import context_reply_chain
from typing_extensions import TypedDict
import json

class AgentState(TypedDict):
    """Agent state representation."""
    email:dict[str,str]
    query:str
    previous_response:str
    generate_email:dict[str,str]
    Context : list[list[str]]

def reply_generator_state(state:AgentState)->AgentState:
    """Generator function to reply based on previous response the email."""
    email = state["email"]
    query = state["query"]
    previous_response = state["previous_response"]
    context = state["Context"]

    response = context_reply_chain.invoke({
        "previous_response":previous_response,
        "email_input":email,
        "query":query,
        "context":context
    })

    return {
        **state,
        "generate_email":response
    }
