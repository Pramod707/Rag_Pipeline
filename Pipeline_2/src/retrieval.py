from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

##embedding the user query
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)
##loading the vector database
vector_db = Chroma(embedding_function=embeddings, persist_directory="db")
##user input
query = input("enter the query :")

res = vector_db.similarity_search(query, k=5)

for i, doc in enumerate(res, 1):
    print("=" * 50)
    print(f"result{i}")
    print(doc.page_content)
