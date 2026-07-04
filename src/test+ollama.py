from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434",
)

print("Embedding...")

vector = embeddings.embed_query("Hello world")

print(len(vector))