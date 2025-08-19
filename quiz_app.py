import streamlit as st
import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from pathlib import Path

st.set_page_config(
    page_title="QuizGPT",
    page_icon = "❓"
)

st.title("QuizGPT")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

@st.cache_data(show_spinner = "Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    Path("./.cache/files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)
    splitter =CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size= 600,
        chunk_overlap= 100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter = splitter)
    return docs

@st.cache_data(show_spinner = "Searching Wikipedia...")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results= 2)
    docs = retriever.get_relevant_documents(topic)
    return docs

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = PromptTemplate.from_template("""
    You are a helpful assistant that is role playing as a teacher.
                    
    Based ONLY on the following context make 5 questions to test the user's knowledge about the text.
    
    Each question should have 3 answers, two of them must be incorrect and one should be correct.

    You can control quiz's Difficulty.
    Each Difficulty have 3 things: EASY, MEDIUM, HARD.
    The Difficulty of Quiz is {level}.

    Context: {context}
""")

@st.cache_data(show_spinner= "Making Quiz...")
def run_quiz_chain(_docs, level):
    chain = prompt | llm
    response = chain.invoke({"context": format_docs(_docs), "level": level})
    r = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(r)


with st.sidebar:
    docs=None
    st.title("Input Line")

    api_key = st.text_input("Put your API key", type="password")

    choice = st.selectbox("Choose what you want to use.", (
        "File","Wikipedia Article",
    ),)
    if choice=="File":
        file = st.file_uploader("Upload a .txt, .pdf or .docx file", type=["pdf","txt","docx"],)
        if file:
            docs = split_file(file)
    else :
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    
    level = st.selectbox("Choose the Level of Quiz.",(
        "EASY","MEDIUM","HARD",
    ))

if not docs:
    st.markdown("""
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """)

else:
    if api_key:
        llm = ChatOpenAI(
            temperature = 1,
            model="gpt-5-nano-2025-08-07",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            api_key=api_key,
        ).bind(
            function_call={"name": "create_quiz"},
            functions=[function],
        )
        response = run_quiz_chain(docs, level)
        question_count = len(response["questions"])
        success_count = 0
        with st.form("quiz_form"):
            
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio("Select an option", [
                    answer["answer"] for answer in question["answers"]
                    ],index=None)
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    success_count+=1
                elif value is not None:
                    st.error("Wrong")
            button = st.form_submit_button()
            st.write(f"Quiz result: {success_count} / {question_count}")
        if(question_count == success_count):
            st.balloons()
    else:
        st.error("You have to put your API key first!")