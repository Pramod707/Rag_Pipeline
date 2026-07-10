from langchain_community.document_loaders  import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
##load file
loader = PyPDFLoader("sl_booklet.pdf")
document = loader.load()
##split the file
splitter = RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap = 200,
)
chunks = splitter.split_documents(document)
##embeddings 
embedded =  OllamaEmbeddings(
    model = "nomic-embed-text"
)
# vectors = embedded.embed_documents(chunks)
##create db 
db = Chroma.from_documents(
    documents = chunks,
embedding= embedded,
    persist_directory = "chroma_db"

)
print(chunks[0])

print("ingestion is done !!")