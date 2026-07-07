from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

docs = [Document(page_content="Hello World")]

db = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory="./db_test"
)

print("Success!")
