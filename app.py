from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st
from time import sleep

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def document_process(path):
    ##document loading
    loader = PyPDFLoader(path)
    docs = loader.load()    

    ##splitting doc 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = splitter.split_documents(docs)

    ##embedding and Vector DB
    embedding = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2-preview")
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding = embedding
    )
    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True

# print(answer.content)
##Streamlit 
st.subheader("Document Q&A ChatBot - Ask Anything")

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

## document upload
if not st.session_state.document_uploaded:
    file = st.file_uploader(label = "Select your PDF File", type="pdf")
    if file:
        with open("document_upload.pdf","wb") as f:
            f.write(file.getvalue())
        
        with st.spinner("Processing Document"):
            document_process("./document_upload.pdf")
        
        st.markdown("Document Uploaded Successfully")
        sleep(2)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.vector_db:

    for oneMessage in st.session_state.messages:
        role = oneMessage["role"]
        content = oneMessage["content"]

        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything")
    if query:

        st.chat_message("user").markdown(query)

        st.session_state.messages.append({"role":"user", "content":query})

        documents = st.session_state.vector_db.similarity_search(query, k= 3)
        context = ""
        
        for doc in documents:
            context += doc.page_content + "\n\n"
        
        prompt = f"""
            You are a helpful assistant that answers questions strictly based on the provided context.

            Context: {context}

            Question:{query}

            Instructions:
            - Answer only using the information in the context above
            - Be concise and to the point
            - Do not make up or assume any information

            Answer:
        """
        result = llm.invoke(prompt)

        st.session_state.messages.append({"role":"ai", "content":result.content})
        st.chat_message("ai").markdown(result.content)