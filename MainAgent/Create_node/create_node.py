import json
from langchain_cohere import ChatCohere
from CreateMemory import store, get_user_id
from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import SystemMessage,ToolMessage,BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import PromptTemplate
from langgraph.graph.message import add_messages
from Create_prompt import prompt
from CreateTools import tools, tools_by_name, search_recall_memories
from langchain_core.messages.utils import get_buffer_string  
from langgraph.graph import END
import tiktoken
import os
from dotenv import load_dotenv
load_dotenv()


llm_model = ChatCohere(
    cohere_api_key=os.getenv("MAIN_COHERE_API_KEY"),
    temperature=0
).bind_tools(
    tools=tools
)

tokenizer = tiktoken.get_encoding("cl100k_base")

class AgentState(TypedDict):
    recall_memories: List[str]
    messages: Annotated[Sequence[BaseMessage], add_messages]

def tool_node(state: AgentState):
    outputs = []
    last_message = state["messages"][-1]
    print(f"last_message:{last_message}")
    if hasattr(last_message, "tool_calls"):
        for tool_call in last_message.tool_calls:
            # print(tool_call["name"])
            # print(tools_by_name.keys())
            # print(tool_call["args"])
            # print("Tool Object:", tools_by_name["model_generation"])
            # print("Has Invoke Method:", hasattr(tools_by_name["model_generation"], "invoke"))
            
            tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            # print(type(tool_result), tool_result)

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call.get("id", ""),
                )
            )
    return {"messages": outputs}

def agent(state: AgentState,config: RunnableConfig) -> AgentState:
    """Process the current state and generate a response using the LLM."""
    messages = state["messages"]
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = llm_model.invoke(
        [prompt] + messages + [recall_str], config
    )
    return {"messages": [prediction]}

def load_memories(state: AgentState, config: RunnableConfig) -> AgentState:
    """Loads relevant memories based on conversation history."""
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke({"comb_str": convo_str, "config": config})
    return {"recall_memories": recall_memories}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
