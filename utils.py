from langchain_google_genai import ChatGoogleGenerativeAI
from state_and_schemas import EvalSchema
from dotenv import load_dotenv
from langchain_core.tools import tool


load_dotenv()

# defining the llm
def get_llm():
    llm=ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )
    return llm

# structured llm for eval
def llm_with_structured_output(schema):
    llm=get_llm()
    return llm.with_structured_output(schema)

def extract_text(content)->str:
    if isinstance(content,str):
        return content
    return "".join(m.get("text","") for m in content if isinstance(m,dict))



    