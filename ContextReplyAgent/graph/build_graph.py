from .build_state import reply_generator_state,AgentState
from langgraph.graph import START, StateGraph, END

def compile_reply_agent():
    print("Compiling Reply Agent Workflow...")
    workflow = StateGraph(AgentState)

    workflow.add_node("Context Reply Agent",reply_generator_state)

    workflow.add_edge(START,"Context Reply Agent")
    workflow.add_edge("Context Reply Agent",END)

    return workflow.compile()