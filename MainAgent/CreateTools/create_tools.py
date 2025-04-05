from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.runnables import RunnableConfig
import uuid
import json
import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CreateMemory import store, get_user_id
from typing import List,Optional
import sys
import os

# Add Refinement_agent folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Refinement_agent')))
from Refinement_agent import compile_refinement_agent
from ContextReplyAgent import compile_reply_agent
from Composing_mail import compile_generate_agent
load_dotenv()

composing_agent = compile_generate_agent()
reply_agent = compile_reply_agent()
refinement_agent = compile_refinement_agent()

from Retriever import retriever

@tool
def Retrieve_data(
    query: str,
    email_sub: Optional[str] = None,
    config: Optional[RunnableConfig] = None
) -> List[List[str]]:
    """
    Retrieves relevant documents based on the input query.
    Optionally takes an email subject and config for further filtering or customization.
    Returns a list of relevant document chunks.
    """
    documents = retriever.get_relevant_documents(query)

    if not documents:
        return [["No relevant content found in memory."]]

    if email_sub:
        documents = [
            doc for doc in documents
            if email_sub.lower() in doc.page_content.lower()
        ]
        if not documents:
            return [[f"No content found related to subject: '{email_sub}'."]]
        
    print("\n[Retrieved Documents]")
    for doc in documents:
        print("-", doc.page_content)

    return [[doc.page_content] for doc in documents]

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    store.add_documents([document])
    return memory


@tool
def search_recall_memories(comb_str: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = store.similarity_search(
        comb_str, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]

@tool
def refinement_tool(email_input:dict[str,str],query:str,config:RunnableConfig)->str:
    """For refining email to convert it to most refined user friendly output"""
    response = refinement_agent.invoke({
        "email":email_input,
        "query":query
    })

    refined_email = response["refined_email"]
    refined_email_str = json.dumps(refined_email)

    save_recall_memory.invoke({
        "memory":refined_email_str,
        "config":config
    })
    return refined_email_str

@tool
def reply_agent_tool(email_input:dict[str,str],query:str,previous_responses:str,config:RunnableConfig)->str:
    """For reply to some email using previous responses on that topic"""
    current_query = query
    output_str = ""
    context = Retrieve_data.invoke({
    "query": current_query,
    "subject": email_input["subject"],
    "config": config
    })
    while True:
        response = reply_agent.invoke({
            "previous_response":previous_responses,
            "email":email_input,
            "query":current_query,
            "context":context
        })

        result = response["generate_email"]
        output_str = refinement_tool.invoke({
            "email_input":result,
            "query":current_query,
            "config" : config}
            )
        
        print("Generated Email:\n", output_str)
        user_satisfied = input("Are you satisfied with this email? (yes/no): ").strip().lower()

        if user_satisfied == "yes":
            break
        elif user_satisfied == "no":
            current_query = input("Enter your updated instruction: ")
        else:
            print("Instruction cannot be empty. Please enter something.")
            continue
    email_input_str = json.dumps(email_input)
    combined_str = email_input_str + output_str + previous_responses
    save_recall_memory.invoke({
        "memory":combined_str
    },config = config
    )
    
    return output_str


@tool
def composing_email_tool(email_input: dict[str, str], query: str, config: RunnableConfig) -> str:
    """Composes email and keeps improving it with user feedback"""
    current_query = query
    output_str = ""
    context = Retrieve_data.invoke({
    "query": current_query,
    "subject": email_input["subject"],
    "config": config
    })
    while True:
        response = composing_agent.invoke({
            "email": email_input,
            "query": current_query,
            "context": context
        })

        result = response["generate_email"]
        output_str = refinement_tool.invoke({
            "email_input": result,
            "query": current_query,
            "config": config
        })

        print("Generated Email:\n", output_str)
        user_satisfied = input("Are you satisfied with this email? (yes/no): ").strip().lower()

        if user_satisfied == "yes":
            break
        elif user_satisfied == "no":
            current_query = input("Enter your updated instruction: ")
        else:
            print("Instruction cannot be empty. Please enter something.")
            continue

    save_recall_memory.invoke(
        {"memory": json.dumps(output_str, indent=2)}, 
        config=config
    )

    return output_str



