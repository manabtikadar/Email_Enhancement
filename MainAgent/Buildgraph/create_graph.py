from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Create_node import AgentState, load_memories, agent, should_continue, tool_node
from langchain_core.runnables import RunnableConfig

def compile_agent():
    print("Compiling Response Agent Workflow...")
    builder = StateGraph(AgentState)
    builder.add_node("load_memories", load_memories)
    builder.add_node("Model_generative_Agent", agent)
    builder.add_node("Agent_tools", tool_node)

    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "Model_generative_Agent")
    builder.add_conditional_edges(
                                  "Model_generative_Agent", 
                                  should_continue, 
                                  {
                                    "continue":"Agent_tools",
                                    "end": END
                                  }
                                  )
    builder.add_edge("Agent_tools", "Model_generative_Agent")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)