from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "./chroma_db"

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

db = Chroma(
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
