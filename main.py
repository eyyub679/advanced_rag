from agent import get_agent_builder

def main():
    agent_builder=get_agent_builder()
    agent=agent_builder.compile()
    user_query=input()
    stream=agent.stream({
        "query":user_query,
        "documents":[],
        "relevance_score":"pending",
        "answer":"",
        "is_hallucination": "pending",
        "hallucination_feedback":[],
        "answer_quality":"pending",
        "answer_quality_feedback": [],
        "n_iterations":0
    },stream_mode="updates")
    answer=""
    for event in stream:
        for node_name,output in event.items():
            print(f"{node_name} completed")
            if "answer" in output:
                answer=output["answer"]
    print(answer)
if __name__ == "__main__":
    main()
