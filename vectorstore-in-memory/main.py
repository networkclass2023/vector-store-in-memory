import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# from langchain.ducument_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain_community.llms import OpenAI
from langchain_openai import OpenAI


if __name__ == "__main__":
    load_dotenv()
    print("Hello VectorStore")
    key = os.environ.get("OPENAI_API_KEY")
    print(key)
    pdf_path = (
        "D:/pr/backend/newc/vetorstore-in-memory/vectorstore-in-memory/2210.03629.pdf"
    )
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    docs=text_splitter.split_documents(documents=documents)
    
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(docs,embeddings)
    vectorstore.save_local("fiass_index_react")
    
    new_vectorstone=FAISS.load_local("fiass_index_react",embeddings)
    qa=RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff",retriever=new_vectorstone.as_retriever())
    res=qa.run("Give me the gist of ReAct in 3 sentences")
    print(res)
    