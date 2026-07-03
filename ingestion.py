from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
embeddings = OllamaEmbeddings(
    model = "mxbai-embed-large"
)
load_dotenv()
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = []

for url in urls:
    loader = WebBaseLoader(url)
    loaded_docs = loader.load()
    docs.append(loaded_docs)

docs_list = []

for sublist in docs:
    for item in sublist:
        docs_list.append(item)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
doc_splits = text_splitter.split_documents(docs_list)
vectorstore = FAISS.from_documents(
    doc_splits,
    embeddings
)

retriever = vectorstore.as_retriever()