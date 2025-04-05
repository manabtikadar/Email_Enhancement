import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MainAgent import compile_agent
from Router import triage_router_chain
from langchain_core.runnables import RunnableConfig
from typing import TypedDict
import json

response_rag_agent = compile_agent()

class GraphState(TypedDict):
    """
    TypedDict for the state of the graph.
    """
    messages_query: str
    reasoning:str
    classification:str
    generated_email:str

def triage_router(state: GraphState) -> GraphState:
    """
    Function to triage the router based on the state.
    """
    messages_query = state["messages_query"]
    triage_router_response = triage_router_chain.invoke({"messages_query": messages_query})

    response = triage_router_response.additional_kwargs["tool_calls"][0]["function"]["arguments"]
    response = json.loads(response)

    return {
        **state,
        "reasoning": response["reasoning"],
        "classification": response["classification"]
    }

def decide_to_triage(state: GraphState) -> str:
    response = state["classification"]
    if response == "Respond":
        print(" Classification: RESPOND - This email requires a response")
        return "response-agent"
    elif response == "Ignore":
        print(" Classification: IGNORE - This email can be safely ignored")
        return "end"
    elif response == "Notify":
        print(" Classification: NOTIFY - This email contains important information")
        return "notify"
    else:
        raise ValueError(f"Invalid classification: {response}")
    
def response_rag_agent_state(state: GraphState,config:RunnableConfig) -> GraphState:
    """
    Function to run the response agent based on the state.
    """
    messages_query = state["messages_query"]
    response = response_rag_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": messages_query
                }
        ]
        },
        config=config
    )
    return {
        **state,
        "generated_email":response["messages"][-1].content
    }

def notify_state(state: GraphState, config: RunnableConfig) -> str:
    """
    Function to notify the user based on the state.
    """
    return f"📢 Notification sent to the user's 📱 phone number assigned to 👤 user ID: {config['configurable']['user_id']} ✅"

def ignore_state(state: GraphState,config:RunnableConfig) -> str:
    """
    Function to ignore the state.
    """
    return "🚫 This email can be safely ignored — it appears to be irrelevant, spam 🗑, or contains personal information 🔒."