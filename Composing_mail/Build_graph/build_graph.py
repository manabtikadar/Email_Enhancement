from langgraph.graph import START, StateGraph, END
from .build_state import (
  generate_email, 
  Email_Type_Router,
  AgentState
)

def compile_generate_agent():
    print("Compiling Composer Agent workflow...")
    workflow = StateGraph(AgentState)

    nodes = {
        "Email Type Router": Email_Type_Router,
        "Generation Node": generate_email,
    }

    for name, func in nodes.items():
        workflow.add_node(name, func)

    workflow.add_edge(START, "Email Type Router")
    workflow.add_edge("Email Type Router", "Generation Node")
    workflow.add_edge("Generation Node", END)

    return workflow.compile()




