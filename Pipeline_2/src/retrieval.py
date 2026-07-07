from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

##embedding the user query
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)
##loading the vector database
vector_db = Chroma(embedding_function=embeddings, persist_directory="db")
##user input and converting user query into vectors
query = input("enter the query :")
embedd_vector = embeddings.embed_query(query)

## converting the user query into vectors and retrieving the top 5 chunks
res = vector_db.similarity_search_by_vector(embedd_vector, k=5)
##generating the results
for i, doc in enumerate(res, 1):
    print("=" * 50)
    print(f"result{i}")
    print(doc.page_content)
