from dotenv import load_dotenv  # Loads environment variables from the .env file without manuely loading
load_dotenv()                   # Reads the .env file and makes its variables available to the program 

from langchain_community.document_loaders import PyPDFLoader                                # Loads and extracts text from PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter                         # Splits large text into smaller overlapping chunks.
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings      # Gemini LLM and embedding model.
from langchain_community.vectorstores import InMemoryVectorStore                            # Stores document embeddings in memory for similarity search.
import streamlit as st                                                                      # Builds the web interface for the chatbot.
from time import sleep                                                                      

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash") 

if "vector_db" not in st.session_state: 
    st.session_state.vector_db = None   # Creates a session variable to store the vector database. Initialized only once per user session to avoid recreation on every rerun.


if "messages" not in st.session_state: # Creates a session variable to store the chat history. Keeps previous user and AI messages persistent across Streamlit reruns
    st.session_state.messages = []

def document_process(path):
    ##document loading
    loader = PyPDFLoader(path)        # Creates a PDF loader object for the given file path.
    docs = loader.load()              # Extracts text from the PDF and stores it as Document objects.  

    ##splitting doc 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200) # Maximum number of characters per chunk. Shares 200 characters between consecutive chunks to preserve context.
    docs = splitter.split_documents(docs)  # Splits the PDF into smaller, overlapping chunks.

    ##embedding and Vector DB
    embedding = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-2-preview") # Converts each text chunk into a numerical embedding (vector).
    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,             # Stores all document chunks.
        embedding = embedding       # Generates embeddings and indexes them for similarity search.
    )
    st.session_state.vector_db = vector_db              # Saves the vector database in the Streamlit session so it can be reused without rebuilding.
    st.session_state.document_uploaded = True           # Sets a flag indicating the PDF has been successfully processed and is ready for querying.

# print(answer.content)
##Streamlit 
st.subheader("Document Q&A ChatBot - Ask Anything")     # Displays the title of the chatbot interface.

if "document_uploaded" not in st.session_state:         # Creates a session variable to track whether a PDF has already been processed. Initialized only once per user session.
    st.session_state.document_uploaded = False

## document upload
if not st.session_state.document_uploaded:
    file = st.file_uploader(label = "Select your PDF File", type="pdf")             # Allows the user to upload only PDF files.
    if file:
        with open("document_upload.pdf","wb") as f:                                 # Saves the uploaded PDF temporarily to the local system.
            f.write(file.getvalue())
        
        with st.spinner("Processing Document"):                                     # Displays a loading spinner while:
            document_process("./document_upload.pdf")                               # 1. Loading the PDF -> Splitting it into chunks -> Creating embeddings ->Building the vector database
        
        st.markdown("Document Uploaded Successfully")
        sleep(2)
        st.rerun()                                                                  # Restarts the Streamlit app so the chatbot interface appears instead of the upload section.

if st.session_state.document_uploaded and st.session_state.vector_db:

    for oneMessage in st.session_state.messages:                                     # Displays each previous message in the chat interface.
        role = oneMessage["role"]
        content = oneMessage["content"]
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything")                                           # Displays a chat input box and waits for the user to ask a question.

    if query:

        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role":"user", "content":query})

        documents = st.session_state.vector_db.similarity_search(query, k= 3)       # Retrieve Relevant Chunks, Finds the 3 document chunks most similar to the user's question.
        context = ""
        
        for doc in documents:
            context += doc.page_content + "\n\n"                                    # Document( page_content="Some chunk of text...", metadata={"page": 5})
        
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

        st.session_state.messages.append({"role":"ai", "content":result.content})           # AIMessage( content="The answer generated by Gemini...",response_metadata={...},usage_metadata={...},additional_kwargs={})
        st.chat_message("ai").markdown(result.content)