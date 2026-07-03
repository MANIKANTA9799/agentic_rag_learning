from langchain_classic import hub 
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(
   model="gpt-oss:20b"
)
prompt = hub.pull(
    'rlm/rag-prompt'
)
gen_chain = prompt|llm|StrOutputParser()
