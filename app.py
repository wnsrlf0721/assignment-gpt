from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

##### Implementation Part #####
answers_prompt = ChatPromptTemplate.from_template("""
Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make up anything up.

Then, give a score to the answer between 0 and 5. 0 being not helpful to the user and 5 being helpful to the user.

Examples:

Question: How far away is the moon?
Answer: The moon is 384,400 km away.
Score: 5

Question: How far away is the sun?
Answer: I don't know
Score: 0

Your turn!

Context: {context}
Question: {question}
""")

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    answers = [
        {
            "answer": answers_chain.invoke(
            {"question": question, "context": doc.page_content}
        ).content,
        "source": doc.metadata["source"],
        "date": doc.metadata["lastmod"],
        }
        for doc in docs
    ]
    return {
        "question": question,
        "answers": answers
    }

choose_prompt = ChatPromptTemplate.from_messages([
    ("system","""
    Use ONLY the following pre-existing answers to answer the user's question.

    Use the answers that have the highest score (more helpful) and favor the most recent ones.

    Return ONLY the answer text, not return about source and date!

    Answers: {answers}
    """),
    ("human","{question}"),
])

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"Answer:{answer['answer']}" 
        for answer in answers)
    
    return choose_chain.invoke({
        "question": question,
        "answers": condensed
    })

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

def parse_page(soup):
    header = soup.find("header")
    nav = soup.find("nav")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if nav:
        nav.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
    )

url = "https://developers.cloudflare.com/sitemap-0.xml"
@st.cache_data(show_spinner="Loading Pre-required Infomation...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^https://developers\.cloudflare\.com/ai-gateway/.*", #AI Gateway Docs
            r"^https://developers\.cloudflare\.com/vectorize/.*", #Cloudflare Vectorize Docs
            r"^https://developers\.cloudflare\.com/workers-ai/$" #Workers AI Docs
        ],
        parsing_function = parse_page
    )
    loader.requests_per_second = 5
    
    docs = loader.load_and_split(text_splitter = splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(api_key = api_key,))
    return vector_store.as_retriever()

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

##### End Implementation ##### 

##### Screen Part #####
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️"
)

st.title("SiteGPT")
st.markdown("""
Ask questions about the documnet of Cloudflare.

Writing your OPENAI API key First.

If you ask a question about Cloudflare, you can get answers By ChatAI.
""")

with st.sidebar:
    api_key = st.text_input("Put your API key", type="password")
    st.markdown("----")
    st.write("Github: https://github.com/wnsrlf0721/assignment-gpt")

retriever = load_website(url)

if not api_key:
    st.error("Please input your OpenAI API Key on the sidebar, then you ask questions")
    st.session_state["messages"]=[]
else:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming= True,
        api_key = api_key,
        callbacks=[
            ChatCallbackHandler(),
        ]
    )
    chain = {
        "docs": retriever, 
        "question": RunnablePassthrough(),
    } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

    send_message("I'm ready! Ask away!", "ai",save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        result = chain.invoke(message)
        with st.chat_message("ai"):
            st.write(result.content)
##### End Screen #####