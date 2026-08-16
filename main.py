from dotenv import load_dotenv
load_dotenv()
import os
from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
import tiktoken
import uuid


chat_history = []  

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

app = FastAPI()
running_summary = ""                                                    # compressed memory of "old" turns
recent_history = []                                                     # verbatim recent turns, not yet summarized
TOKEN_BUDGET = 150
SUMMARIZE_CHUNK = 4                                                     # summarize this many oldest messages at a time when over budget

class Query(BaseModel):                                                 # Defines the expected JSON request body with a required string field named 'question'.
    question : str
    doc_id : str

def get_vector_db(doc_id : str):
    persist_dir = f"chroma_db/{doc_id}"
    pdf_path = f"uploads/{doc_id}.pdf"

    if not os.path.exists(persist_dir) or not os.path.exists(pdf_path):
        return None
    embedding = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2-preview")
    return Chroma(persist_directory=persist_dir, embedding_function=embedding)

def format_history(history):
    formatted = ""
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        formatted += f"{role}:{turn['content']}\n"
    return formatted

encoder = tiktoken.get_encoding("cl100k_base")

def count_token(text):
    return len(encoder.encode(text))                                    # encoder.encode(text) turns a string into a list of token IDs; len(...) gives you the count

def history_token_count(history):
    text = format_history(history)
    return count_token(text)

def summarize_turns(old_summary, turns_to_summary):
    turns_text = format_history(turns_to_summary)

    prompt = f"""
        You are maintaining a running summary of a conversation.

        Existing summary so far:
        {old_summary if old_summary else "(none yet)"}

        New turns to fold in:
        {turns_text}

        Instructions:
        - Merge the new turns into the existing summary
        - Keep only key facts, decisions, and topics discussed
        - Be concise — 3-5 sentences maximum
        - Do not lose important details from the existing summary

        Updated summary:
    """

    result = llm.invoke(prompt)
    return result.content

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):         # Declares a required file upload parameter. FastAPI expects a file named 'file' in multipart/form-data, automatically converts it into an UploadFile object, and passes it to the function.

    doc_id = str(uuid.uuid4())                              # generate 128-bit universally unique identifier
    file_path = f"uploads/{doc_id}.pdf"
    os.makedirs("uploads",exist_ok=True)                    # create a folder and exist_ok=True ignore any errors if the target folder already exists
    
    # 1. save uploaded bytes to disk — same idea as your Streamlit code
    contents = await file.read()
    with open(file_path,"wb") as f:
        f.write(contents)  

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    persist_dir = f"chroma_db/{doc_id}"
    vector_db = Chroma.from_documents(              #vector_db is a Chroma vector-store object. It provides an interface to your stored document chunks and their embeddings
        documents = docs,
        embedding = embedding,
        persist_directory= persist_dir
    ) 


    return {"message": "Document uploaded and processed successfully", "doc_id":doc_id}

@app.post("/ask")
async def ask_question(query: Query):

    global running_summary, recent_history

    vector_db = get_vector_db(query.doc_id)

    if vector_db is None:
        return {"error": "No document uploaded yet. Upload a PDF first."}

    document = vector_db.similarity_search(query.question, k=3)
    context = ""
    for doc in document:
        context += doc.page_content + "\n\n"

    prompt = f"""                      
        You are a helpful assistant that answers questions strictly based on the provided context.
        
        Summary of earlier conversation:
        {running_summary if running_summary else "(none yet)"}   

        Recent conversation:
        {format_history(recent_history)}

        Context: {context}

        Question: {query.question}

        Instructions:
        - Answer only using the information in the context above
        - Be concise and to the point
        - Do not make up or assume any information

        Answer:
    """

    result = llm.invoke(prompt)
    recent_history.append({"role":"user","content":query.question})
    recent_history.append({"role":"ai","content":result.content})

    # check if recent_history has grown past budget
    if history_token_count(recent_history) > TOKEN_BUDGET:
        turns_to_summarize = recent_history[:SUMMARIZE_CHUNK]
        recent_history = recent_history[SUMMARIZE_CHUNK:]
        running_summary = summarize_turns(running_summary, turns_to_summarize)
        print(f"\n[SUMMARIZED] New running_summary:\n{running_summary}\n")

    print(f"[STATE] recent_history has {len(recent_history)} messages, {history_token_count(recent_history)} tokens\n")
    return {"answer": result.content}