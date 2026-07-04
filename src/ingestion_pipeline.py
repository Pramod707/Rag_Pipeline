import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import shutil

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


def create_vector_store(chunks):
    print("Creating embeddings and vector store...")

    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )

    persist_dir = "./chroma_db"

    # Remove old database (optional while developing)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    vector_store = Chroma(
        collection_name="rag_collection",
        embedding_function=embedding_model,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

    batch_size = 50

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        print(
            f"Adding batch {i // batch_size + 1} "
            f"({i} -> {min(i + batch_size, len(chunks))})"
        )

        vector_store.add_documents(batch)

    print("Vector store created successfully!")

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
