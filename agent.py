from rag.retriever import retrieve_documents
from langgraph.graph import StateGraph, START, END
from state_and_schemas import State
from utils import llm_with_structured_output,get_llm,extract_text
from typing import cast
from state_and_schemas import EvalSchema,HallucinationSchema,AnswerQualitySchema,RouterSchema
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langsmith import Client
from rag.reranker import rerank

load_dotenv()

client=TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)

hub_client=Client()

def get_agent_builder():
    # hyphotetical document embeddings
    def hyde(state:State)->dict:
        """Produces a hyphotetical document for a better retrieval"""
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a movie encyclopedia.
                    Write a short,factual synopsis based on the following question.
                    Write it in the style of a TMDB movie overview.
                    Do not include any premable, explanation or disclaimers.
                    Just give the synopsis directly
                    """
                ),
                (
                    "human",
                    "Question: {question}"
                )
            ]
        )
        llm=get_llm()
        chain=prompt|llm
        query=state["query"]
        response=chain.invoke({"question":query})
        return {"query":f"{query}{extract_text(response.content)}"}
    # decides which way to go
    def route_after_question(state:State)->str:
        """Based on user's query decides which path to choose"""
        llm=llm_with_structured_output(RouterSchema)
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert router assistant.
                    Read user's query and decide which route to choose.

                    If user's query is not about ROMANCE movies, recommendations, or any information about the plot of a movie:
                        Output "out_of_bound".
                    If user's query is about movies but does not require advanced search:
                        Output "simple_rag"
                    If user's query is about movies and requires multiple searches and advanced rag:
                        Output "complex_rag"
                    Else:
                        Output "general_conversation" (which can be answered with an llm's existing knowledge)
                    """
                ),
                (
                    "human",
                    "User's Query: {query}"
                )
            ]
        )
        chain=prompt|llm
        response=cast(RouterSchema,chain.invoke({
            "query":state["query"]
        }))
        route=response.route
        if route=="out_of_bound":
            return "generator_out_of_bound"
        elif route=="simple_rag":
            return "retriever_simple"
        elif route=="general_conversation":
            return "generator_general"
        else:
            return "hyde" # our main graph is based on complex geenration so we dont change the names

    #generator for out_of_bound route
    def generator_out_of_bound(state:State)->dict:
        """Politely answers user's query when it is out_of_bound"""
        response="""
        Sorry, I am an assitant which knows about romance movies, recommendations and other related things.
        Try asking a question about those things instead."""
        return {
            "answer":response
        }
    #generator for general_conversations
    def generator_general(state:State)->dict:
        """LLM handles user's query based on its internal knowledge"""
        llm=get_llm()
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Answer user's query"
                ),
                (
                    "human",
                    "Query: {query}"
                )
            ]
        )
        chain=prompt|llm
        response=chain.invoke({"query":state["query"]})
        return {
            "answer":extract_text(response.content)
        }
    def retriever_simple(state: State)->dict:
        """Retrieves relevant documents for the query for simple_rag"""
        query=state["query"]
        docs=retrieve_documents(query)
        return {"documents":docs}
    def generator_simple_rag(state:State)->dict:
        """Generates answer based on retrieved documents"""
        content="\n\n".join(
            doc.page_content for doc in state["documents"]
        )
        llm=get_llm()
        prompt=hub_client.pull_prompt("rlm/rag-prompt",dangerously_pull_public_prompt=True)
        chain=prompt|llm
        response=chain.invoke(
            {
                "question":state["query"],
                "context":content
            }
        )
        return {
            "answer":extract_text(response.content)
        }

    # initializing the retriever node
    def retriever(state: State)->dict:
        """Retrieves relevant documents for the query"""
        query=state["query"]
        docs=retrieve_documents(query)
        return {"documents":docs}

    def reranker(state:State)->dict:
        """reranks retrieved documents"""
        documents=state["documents"]
        query=state["query"]
        relevant_docs=rerank(documents,query)
        return {"documents":relevant_docs}
    # evaluates retrieved documents
    def evaluator(state: State)->dict:
        """Evaluates relevance of the documents"""
        llm=llm_with_structured_output(EvalSchema)
        content="\n\n".join(
            doc.page_content for doc in state["documents"]
        )
        prompt=ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a strict retrieval quality evaluator.

                Analyze the given question and provided content.

                Output "Correct":
                    If the content is relevant to the question and there is no need for further research.
                Output "Ambigious":
                    If the content is ambigious (it is still somewhat relevant to the question but web search is still needed).
                If neither:
                    Output "Incorrect"
                """
            ),
            (
                "human",
                """
                Question: {question},
                Content: {content}
                """
            )
        ]
        )
        chain=prompt|llm
        response=cast(EvalSchema,chain.invoke(
            {
                "question":state["query"],
                "content":content
            }
        ))
        return {
            "relevance_score":response.relevance_score
        }

    # router function
    def route_after_eval(state:State)->str:
        if state["relevance_score"]=="correct":
            return "generator"
        return "query_rewriter"

    #query rewriter
    def query_rewriter(state:State)->dict:
        """Rewriter query to make it better for web search"""
        llm=get_llm()
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a professional query rewriter.

                    Your task is to rewrite the user's query to make it better for web search.

                    Return ONLY the rewritten query — a single search-ready phrase.
                    No explanations, no bullet points, no options, no quotes, no markdown.
                    """
                ),
                (
                    "human",
                    """Query: {query}"""
                )
            ]
        )
        chain=prompt|llm
        response=chain.invoke(
            {
                "query":state["query"]
            }
        )
        return {
            "query":extract_text(response.content)
        }

    # web search
    def web_search(state: State)->dict:
        """Searches the web to find better results for refined query"""
        response=client.search(state["query"])
        docs=[
            Document(
                page_content=result["content"],
                metadata={
                    "title":result["title"],
                    "url":result["url"]
                }
            )
            for result in response["results"]
        ]
        if state["relevance_score"]=="incorrect":
            return {
                "documents":docs
            }
        return {
            "documents":state["documents"]+docs
        }
    # generator
    def generator(state:State)->dict:
        """Generates answer from given content"""
        llm=get_llm()
        content="\n\n".join(
            doc.page_content for doc in state["documents"]
        )
        messages=[
        (
            "system",
            """
            You are a helpful rag assistant. Answer the user's query based on given content.
            If content does not contain enough information, say you don't know.
            If you see any feedback and score (yes/no or good/bad) about hallucination or answer quality, try to address tne issues and generate better a answer 
            """
        ),
        (
            "user",
            """
            Query: {query}
            Content: {content}
            """
        )
        ]
        prompt_vars={
                        "query":state["query"],
                        "content":content
                    }
        if state["is_hallucination"]=="yes":
            messages.append((
                "human",
                """HALLUCINATION_SCORE: {score}\nHALLUCINATION FEEDBACK: {feedback}"""
            ))
            prompt_vars={
                            "query":state["query"],
                            "content":content,
                            "score":state["is_hallucination"],
                            "feedback":state["hallucination_feedback"]
                        }
        elif state["answer_quality"]=="bad":
            messages.append((
                "human",
                """ANSWER_QUALITY_SCORE: {score}\nANSWER_QUALITY FEEDBACK: {feedback}"""
            ))
            prompt_vars={
                            "query":state["query"],
                            "content":content,
                            "score":state["answer_quality"],
                            "feedback":state["answer_quality_feedback"]
                        }
        prompt=ChatPromptTemplate.from_messages(messages)
        chain=prompt|llm
        response=chain.invoke(prompt_vars)
        return {
            "answer":extract_text(response.content),
            "n_iterations":state["n_iterations"]+1
        }
    #hallucination_detector
    def hallucination_detector(state:State)->dict:
        """Determines if the answer contains hallucinations"""
        content=[doc.page_content for doc in state["documents"]]
        prompt=ChatPromptTemplate([
            (
                "system",
                """
                You are a strict hallucination detector. Analyze the user query, provided content and the generated answer, and output either "yes" or "no".
                "yes" means answer does contain hallucinations and not grounded.
                """
            ),
            (
                "human",
                """Query: {query}\nContent: {content}\nAnswer: {answer}"""
            )
        ])
        llm=llm_with_structured_output(HallucinationSchema)
        chain=prompt|llm
        response=cast(HallucinationSchema,chain.invoke({
            "query":state["query"],
            "content":content,
            "answer":state["answer"]
        }))
        return {
            "is_hallucination":response.grade,
            "hallucination_feedback":response.feedback
        }
    
    #answer quality analyzer
    def answer_quality_analyzer(state:State)->dict:
            """Determines if the answer is relevant to question"""
            prompt=ChatPromptTemplate([
                (
                    "system",
                    """
                    You are a strict answer-quality analyzer. Analyze the user query, and the generated answer to output either "good" or "bad".
                    "good" means answer successfully satisfies the query.
                    """
                ),
                (
                    "human",
                    """Query: {query}\nAnswer: {answer}"""
                )
            ])
            llm=llm_with_structured_output(AnswerQualitySchema)
            chain=prompt|llm
            response=cast(AnswerQualitySchema,chain.invoke({
                "query":state["query"],
                "answer":state["answer"]
            }))
            return {
                "answer_quality":response.grade,
                "answer_quality_feedback":response.feedback
            }
    # router after hallucination detector
    def route_after_answer(state:State)->str:
        """Decides where to go based on answer quality score"""
        if state["answer_quality"]=="bad" and state["n_iterations"]<3:
            return "generator"
        return END
    # router after hallucination detector
    def route_after_hallucination(state:State)->str:
        """Decides where to go based on hallucination score"""
        if state["is_hallucination"]=="yes" and state["n_iterations"]<3:
            return "generator"
        return "answer_quality_analyzer"

    # defining the agent builder
    agent_builder=StateGraph(State)

    # adding and connecting nodes
    agent_builder.add_node("retriever",retriever)
    agent_builder.add_node("hyde",hyde)
    agent_builder.add_node("reranker",reranker)
    agent_builder.add_node("evaluator",evaluator)
    agent_builder.add_node("query_rewriter",query_rewriter)
    agent_builder.add_node("generator_out_of_bound",generator_out_of_bound)
    agent_builder.add_node("generator_general",generator_general)
    agent_builder.add_node("retriever_simple",retriever_simple)
    agent_builder.add_node("generator_simple_rag",generator_simple_rag)   
    agent_builder.add_node("hallucination_detector",hallucination_detector)
    agent_builder.add_node("answer_quality_analyzer",answer_quality_analyzer)
    agent_builder.add_node("web_search",web_search)
    agent_builder.add_node("generator",generator)
    agent_builder.add_conditional_edges(
        START,
        route_after_question,
        ["generator_out_of_bound","retriever_simple","generator_general","hyde"]
    )
    agent_builder.add_edge("hyde","retriever")
    agent_builder.add_edge("generator_out_of_bound",END)
    agent_builder.add_edge("retriever","reranker")
    agent_builder.add_edge("reranker","evaluator")
    agent_builder.add_edge("generator_general",END)
    agent_builder.add_edge("retriever_simple","generator_simple_rag")
    agent_builder.add_edge("generator_simple_rag",END)
    agent_builder.add_edge("query_rewriter","web_search")
    agent_builder.add_edge("web_search","generator")
    agent_builder.add_conditional_edges(
        "evaluator",
        route_after_eval,
        ["generator","query_rewriter"]
    )
    agent_builder.add_edge("generator","hallucination_detector")
    agent_builder.add_conditional_edges(
        "hallucination_detector",
        route_after_hallucination,
        ["generator","answer_quality_analyzer"]
    )
    agent_builder.add_conditional_edges(
        "answer_quality_analyzer",
        route_after_answer,
        ["generator",END]
    )
# a quick note: even though i could register the same hyde function as two separate nodes and use them for both simple and complex retrieval, i wont do that cuz simple rag is for speed and simplicity

    return agent_builder