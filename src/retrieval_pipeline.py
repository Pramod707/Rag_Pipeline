from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv

load_dotenv()

persistent_directory = "./chroma_db"

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://127.0.0.1:11434"
)

db = Chroma(
    collection_name="rag_collection",
    embedding_function=embedding_model,
    persist_directory=persistent_directory,
    collection_metadata={"hnsw:space": "cosine"},
)

query = "which island does spaceX lease for its launches in pacific?"

retriever = db.as_retriever(search_kwargs={"k": 5})


relavent_doc = retriever.invoke(query)

print(f"user query {query}")

for i, doc in enumerate(relavent_doc):
    print(f"\nDocument{i}:\n{doc.page_content}\n")
