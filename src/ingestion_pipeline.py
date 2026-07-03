import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import embeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_files(doc_path="docs"):
    """ " load all the txt files"""
    print(f"loading files from{doc_path}")
    if not os.path.exists(doc_path):
        raise FileNotFoundError("the path does not exist!!")

    loader = DirectoryLoader(path=doc_path, glob="*.txt", loader_cls=TextLoader)

    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"no text files found in path{doc_path}")

    for i, documents in enumerate(documents[:2]):
        print(f"\nDocument{i + 1}")
        print(f"source: {documents.metadata['source']}")
        print(f"content len : {len(documents.page_content)} characters")
        print(f"metadata: {documents.metadata}")
    return documents


# def main():

#     #load the files
#     #chunk the files
#     #store in chroma db
