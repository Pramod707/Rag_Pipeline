from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory="db",
)

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.1,
)
query = input("enter the query : ")
embed_query = embeddings.embed_query(query)
docs = vector_db.similarity_search_by_vector(embedding=embed_query, k=3)

context = "\n\n".join(doc.page_content for doc in docs)

prompt = f"""
Answer the question using only  the context Below.

Context:
{context}

question:
{query}
"""
response = llm.invoke(prompt)

print(response)
