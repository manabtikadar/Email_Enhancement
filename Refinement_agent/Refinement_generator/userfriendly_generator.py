from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
    You are an expert email assistant dedicated to refining and improving email responses. 

    **Original Email:**  
    {email}  

    **List of Improvements Needed:**  
    {ImprovementList}  

    ### Task:  
    Improve the given email while maintaining professionalism and clarity.  
    Ensure the output is returned in **strict JSON format** as follows:  

    ```json
    {{
        "From": "<Original Sender from user's email input>",
        "To": "<Original Recipient from user's email input>",
        "Subject": "<Refined Subject Line If needed>",
        "Body": "<Refined Email Body with Improvements>"
    }}
    ```  

    **Rules:**  
    - Keep the response formal and professional.  
    - Preserve the original intent while enhancing grammar, clarity, and tone.  
    - Ensure the email structure remains intact.  
    - Return only the JSON output without any extra text.  
    """,
    input_variables=["email", "ImprovementList"]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0
)

email_refiner_chain = (
    prompt 
    | llm 
    | JsonOutputParser()
)