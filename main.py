from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

app = FastAPI()
vector_db = None                                            # global state — lives as long as the server process runs

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):         # Declares a required file upload parameter. 
    # FastAPI expects a file named 'file' in multipart/form-data, automatically converts it into an UploadFile object, and passes it to the function.
    global vector_db    # Refers to the module-level vector_db so changes inside the function persist globally.

    # 1. save uploaded bytes to disk — same idea as your Streamlit code
    contents = await file.read()
    with open("document_upload.pdf","wb") as f:
        f.write(contents)  

    loader = PyPDFLoader("document_upload.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(documents=docs, embedding=embedding)

    return {"message": "Document uploaded and processed successfully"}