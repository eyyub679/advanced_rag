from typing import TypedDict,Literal, Annotated
from langchain_core.documents import Document
from pydantic import BaseModel,Field
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

# state of the graph
class State(TypedDict):
    """
    State of the graph

    Attributes:
        query: user's query (str)
        documents: retrieved documents (a list of strings)
        relevance_score: evaluator's response (correct,incorrect,ambigious)
        answer: answer to the user query (string)
        is_hallucination: whether the generated response is a hallucination (yes,no)
        hallucination_feedback: flaws that cause hallucination (list of strings)
        answer_quality: whether the answer quality is bad or good
        answer_quality_feedback: flaws that lead to a bad answer (list of strings)
        n_iterations: max number of iterations (int)
    """
    query: str
    documents: list[Document]
    relevance_score: Literal["correct","incorrect","ambigious","pending"]
    answer: str
    is_hallucination: Literal["yes","no","pending"]
    hallucination_feedback: list[str]
    answer_quality: Literal["good","bad","pending"]
    answer_quality_feedback: list[str]
    n_iterations: int

# schema for evaluator
class EvalSchema(BaseModel):
    """Structure of the eval node's output"""
    relevance_score: Literal["correct","incorrect","ambigious"]=Field(
        ...,
        description="shows how relevant documents are to the question"
    )

class HallucinationSchema(BaseModel): # this is for hallucination detector
    """Structure of hallucination detector"""
    reasoning: str=Field(...,description="LLM's chain of thoughts")
    grade: Literal["yes","no"]=Field(...,description="Binary score of whether the model hallucinated or not")
    feedback: list[str]=Field(...,description="Feedback on flaws")

class AnswerQualitySchema(BaseModel): # this is for answer-quality analyzer
    """Structure of answer-quality analyzer"""
    reasoning: str=Field(...,description="LLM's chain of thoughts")
    grade: Literal["good","bad"]=Field(...,description="Quality of the generated answer")
    feedback: list[str]=Field(...,description="Feedback on flaws")   

class RouterSchema(BaseModel):
    """Router's schema"""
    route: Literal["out_of_bound","simple_rag","complex_rag","general_conversation"]=\
    Field(
        ...,
        description="the route to choose after router's decision"
    )
