from typing import Any ,Dict 
from graph.chains.generation import gen_chain
from graph.state import GraphState


def generate(state:GraphState):
    print("Generate")
    question = state["question"]
    documents = state["documents"]
    gen = gen_chain.invoke({
        "context":documents, "question":question
    })
    return {
        "documents":documents , "question ":question , 
        "generation":gen
    }