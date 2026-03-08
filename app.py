import os
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

# 1. Provide a list of YouTube Video URLs
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=Ng_9Yjd-gPo",
    "https://www.youtube.com/watch?v=CFZWI8ExfuY"
]


def _get_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com") and parsed.path == "/watch":
        return parse_qs(parsed.query).get("v", [None])[0]
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/") or None
    return None


def load_youtube_transcript(url: str, languages: Optional[List[str]] = None) -> List[Document]:
    """Load transcript from a YouTube URL using youtube-transcript-api only (no pytube)."""
    video_id = _get_video_id(url)
    if not video_id:
        raise ValueError(f"Could not determine video ID for URL: {url!r}")

    languages = languages or ["en", "hi"]

    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id, languages=languages)
        transcript_list = list(transcript)
    except Exception:
        try:
            ytt = YouTubeTranscriptApi()
            transcript = ytt.fetch(video_id)
            transcript_list = list(transcript)
        except Exception:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

    text = " ".join(item["text"] if isinstance(item, dict) else item.text for item in transcript_list)

    return [Document(page_content=text, metadata={"source": url, "video_id": video_id})]


def build_knowledge_base():
    all_docs = []
    for url in YOUTUBE_URLS:
        print(f"Loading transcript for: {url}")
        all_docs.extend(load_youtube_transcript(url, languages=["hi", "en"]))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(docs)}")

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    return vectorstore


# Updated prompt - always answer in English
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the context provided below.
The context may be in Hindi or English, but ALWAYS respond in English.
Translate any Hindi content to English in your answer.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
""")

# Initialize Bot
vector_db = build_knowledge_base()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Retriever
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Queries - all answers will be in English
print("\n=== QUERY 1 ===")
response1 = qa_chain.invoke("Summarize everything you know from the videos")
print(response1)

print("\n=== QUERY 2 ===")
response2 = qa_chain.invoke("What are the main topics discussed in the videos?")
print(response2)

print("\n=== QUERY 3 ===")
response3 = qa_chain.invoke("इस वीडियो में क्या बताया गया है?")
print(response3)
