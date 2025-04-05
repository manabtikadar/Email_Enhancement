from .build_state import (
    AgentState,
    refinement_grader, 
    user_friendly_generator
)
from langgraph.graph import START, StateGraph, END

def compile_refinement_agent():
    print("Compiling Refinement Agent workflow...")
    workflow = StateGraph(AgentState)

    nodes = {
        "Refinement Grader": refinement_grader,
        "Refined Email Generate": user_friendly_generator
    }

    for name, func in nodes.items():
        workflow.add_node(name, func)

    workflow.add_edge(START, "Refinement Grader")
    workflow.add_edge("Refinement Grader", "Refined Email Generate")
    workflow.add_edge("Refined Email Generate", END)

    return workflow.compile()