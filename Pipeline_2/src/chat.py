from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory="db",
)

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.1,
)

while True:
    query = input("enter the query : ")
    if query == "exit":
        break
    else:
        #embed_query = embeddings.embed_query(query)
        # docs = vector_db.similarity_search_by_vector(embedding=embed_query, k=3)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        # docs = retriever.invoke(query)
        # print(docs)

        def get_context(docs):
            context = "\n\n".join(doc.page_content for doc in docs)
            return context

        # prompt = f"""
        #     Answer the question using only  the context Below.

        #     Context:
        #     {context}

        #     question:
        #     {query}
        #     """
        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful Ai assistant
            Answer the questions only from given context below
            If the Answer is not present. say  : "I dont know !!"
            context: {context}
            question:{question} 
            
            """
        )
        chain = (
            {
                "context": retriever | get_context,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(query)

        print(response)
