import os
import tempfile
import whisper
from pytube import YouTube
from dotenv import load_dotenv

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi

import gradio as gr

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
parser = StrOutputParser()

template = """
Answer the question based on the context below. If you can't answer, say "I don't know".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# ----------------------------
# BUILD RAG FUNCTION
# ----------------------------
def build_rag(youtube_link):

    TRANSCRIPT_FILE = "transcription.txt"

    # -------------------------
    # 1. Extract video ID
    # -------------------------
    video_id = extract_video_id(youtube_link)

    # -------------------------
    # 2. Fetch transcript (NEW METHOD)
    # -------------------------
    if not os.path.exists(TRANSCRIPT_FILE):

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=["en"])

        # join only text
        text = " ".join([snippet.text for snippet in transcript.snippets])

        with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(text)

    # 1. Download + transcribe only if needed
    # if not os.path.exists("transcription.txt"):
    #     extract_video_id(youtube_link)


    #     youtube = YouTube(youtube_url)
    #     audio = youtube.streams.filter(only_audio=True).first()

    #     whisper_model = whisper.load_model("base")

    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         file_path = audio.download(output_path=tmpdir)
    #         transcription = whisper_model.transcribe(file_path, fp16=False)["text"].strip()

    #     with open("transcription.txt", "w", encoding="utf-8") as f:
    #         f.write(transcription)

    # 2. Load text
    loader = TextLoader("transcription.txt")
    text_documents = loader.load()

    # 3. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = splitter.split_documents(text_documents)

    # 4. Embeddings
    embeddings = OpenAIEmbeddings()

    # 5. Vector DB (local for simplicity)
    vectorstore = DocArrayInMemorySearch.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever()

    # 6. Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | parser
    )

    return chain


# ----------------------------
# GLOBAL CACHE (important)
# ----------------------------
chain_cache = None


# ----------------------------
# GRADIO FUNCTION
# ----------------------------
def qa_pipeline(youtube_link, question):
    global chain_cache

    # rebuild only if new video
    if chain_cache is None:
        chain_cache = build_rag(youtube_link)

    video_id = extract_video_id(youtube_link)
    
    if not video_id:
        return None, "Invalid YouTube URL"

    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

    result = chain_cache.invoke(question)

    return thumbnail_url ,result




def extract_video_id(url):
    # supports:
    # https://youtu.be/VIDEO_ID
    # https://www.youtube.com/watch?v=VIDEO_ID
    # also handles extra params like ?si=

    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    
    if match:
        return match.group(1)
    return None


def process_video(youtube_link, question):
    
    video_id = extract_video_id(youtube_link)
    
    if not video_id:
        return None, "Invalid YouTube URL"

    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

    answer = f"Question received: {question} (placeholder answer)"

    return thumbnail_url, answer


# ----------------------------
# GRADIO UI
# ----------------------------
demo = gr.Interface(
    fn=qa_pipeline,
    inputs=[
        gr.Textbox(label="YouTube Link"),
        gr.Textbox(label="Question")
    ],
    outputs=[
        gr.Image(label="Thumbnail"),
        gr.Textbox(label="Answer")
    ],
    title="YouTube RAG QA System"
)

demo.launch()