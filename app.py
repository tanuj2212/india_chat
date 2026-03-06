import os
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

# 1. Provide a list of YouTube Video URLs
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=VIDEO_ID_1", # Example Dhruv Rathee video
    "https://www.youtube.com/watch?v=VIDEO_ID_2"  # Example Vijeta Dahiya video
]

def build_knowledge_base():
    all_docs = []
    for url in YOUTUBE_URLS:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language=['hi', 'en'])
        all_docs.extend(loader.load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)

    # Create Vector Store (Persisted locally)
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    return vectorstore

# Initialize Bot
vector_db = build_knowledge_base()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())

# Example query
# response = qa_chain.invoke("What did Dhruv say about the Indian economy?")