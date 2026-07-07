# from langchain_community.document_loaders import DirectoryLoader, TextLoader
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


##load file

loader = DirectoryLoader(
    path="../docs2",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = loader.load()

print(f"Loaded {len(documents)} documents\n")

##split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents)

##create embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://127.0.0.1:11434"
)

##create vector database
vector_store = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory="db"
)
# batch_size = 100
# for i in range(0, len(chunks), batch_size):
#     batch = chunks[i : i + batch_size]
#     vector_store.add_documents(batch)

print("embedding stored successfully")
