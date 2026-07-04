import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


##load files function
def load_files(doc_path="docs"):
    """ " load all the txt files"""
    print(f"loading files from{doc_path}")
    if not os.path.exists(doc_path):
        raise FileNotFoundError("the path does not exist!!")

    loader = DirectoryLoader(
        path=doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"no text files found in path{doc_path}")

    for i, document in enumerate(documents[:2]):
        print(f"\nDocument{i + 1}")
        print(f"source: {document.metadata['source']}")
        print(f"content len : {len(document.page_content)} characters")
        print(f"metadata: {document.metadata}")
    return documents


##chunking function
def split_doc(documents, chunk_size=1000, chunk_overlap=200):
    """ "split the doc into smaller chunks"""
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--chunk--{i + 1}")
            print(f"source : {chunk.metadata['source']}")
            print(f"content len : {len(chunk.page_content)} characters")
            print(f"metadata: {chunk.metadata}")
            print(f"content:")
            print(chunk.page_content)
            print("-" * 50)
        if len(chunks) > 5:
            print(f"\n...add {len(chunks) - 5} more chunks...")
    return chunks


##convert into vecotors and store into chroma db


def create_vector_store(
    chunks,
):
    """creating persist vector store"""
    print("creating embeddings and storing in chroma db")

    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    # embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    print("creating vector store")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("vector store created")
    print(f"vector store is created at {vector_store.persist_directory}")
    return vector_store


def main():
    # load the files
    documents = load_files(doc_path="../docs")
    # chunk the files
    chunk = split_doc(documents)
    # store in chroma db
    vector_store = create_vector_store(chunk)


if __name__ == "__main__":
    main()
