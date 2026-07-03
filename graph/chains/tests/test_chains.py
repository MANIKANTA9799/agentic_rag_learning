#whenever we use pytest it only checks for files 
# starting and having test as first name 

from dotenv import load_dotenv
load_dotenv()
from graph.chains.retreival_grader import GradeDocuments,retrieval_grader
from ingestion import retriever
from graph.chains.generation import gen_chain
def test_retrival_answer_yes()->None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    res:GradeDocuments = retrieval_grader.invoke(
        {"question":question,"document":doc_txt}
    )#type:ignore 
    assert res.binary_score == "yes"


def test_retrival_grader_no():
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    res:GradeDocuments = retrieval_grader.invoke(
        {"question":"how to make pizza","document":doc_txt}
    )#type:ignore 
    assert res.binary_score == "no"

def test_generation_chain():
    question = "agent-memory "
    docs = retriever.invoke(question)
    generation = gen_chain.invoke({"context":docs,"question":question})
    print(generation)