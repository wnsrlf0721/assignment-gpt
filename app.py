import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="DocumentGPT",
    page_icon = "📃"
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

@st.cache_data(show_spinner = "Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter =CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size= 600,
        chunk_overlap= 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter= splitter)
    embeddings = OpenAIEmbeddings(api_key = api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message,role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"],message["role"],save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system","""
    Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

    Context: {context}
    """),
    MessagesPlaceholder(variable_name = "chat_history"),
    ("human","{question}"),
])

def load_memory(input):
    return memory.load_memory_variables({})["history"]

st.title("DocumentGPT")

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your API Key and files on the sidebar.
""")

with st.sidebar:
    st.title("Input Line")
    api_key = st.text_input("Put your API key", type="password")
    file = st.file_uploader("Upload a .txt, .pdf or .docx file", type=["pdf","txt","docx"],)


if file:
    if not api_key:
        st.error("API Key를 입력하지 않았습니다.")
    else:
        llm = ChatOpenAI(
            temperature=0.1, 
            streaming= True,
            api_key = api_key,
            callbacks=[
                ChatCallbackHandler(),
            ]
        )

        #전체 prompt를 메모리에 저장하는 방식 활용
        memory = ConversationBufferMemory(
            llm = llm,
            return_messages=True
        )

        retriever = embed_file(file)
        send_message("I'm ready! Ask away!", "ai",save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")
            chain = {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "chat_history": load_memory
            } | prompt | llm
            with st.chat_message("ai"):
                result = chain.invoke(message)
                memory.save_context({"question": message},{"output": result.content})
    
else:
    st.session_state["messages"]=[]