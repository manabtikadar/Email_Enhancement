from .build_state import GraphState,triage_router,decide_to_triage,response_rag_agent_state,notify_state,ignore_state
from langgraph.graph import StateGraph, START, END

def compile_full_agent_graph():
    """
    Function to compile the main agent graph.
    """
    builder = StateGraph(GraphState)

    # Define nodes
    builder.add_node("triage_router", triage_router)
    builder.add_node("response_rag_agent", response_rag_agent_state)
    builder.add_node("notify_state", notify_state)
    builder.add_node("ignore_state", ignore_state)

    # Define edges
    builder.add_edge(START, "triage_router")
    builder.add_conditional_edges(
        "triage_router",
        decide_to_triage,
        {
            "response-agent": "response_rag_agent",
            "notify": "notify_state",
            "end": "ignore_state"
        }
    )
    builder.add_edge("response_rag_agent", END)
    builder.add_edge("ignore_state", END)
    builder.add_edge("notify_state", END)

    return builder.compile()