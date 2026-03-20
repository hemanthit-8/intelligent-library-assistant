import os
import re
import streamlit as st
from typing import TypedDict, Annotated
from operator import add

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

os.environ["TOKENIZERS_PARALLELISM"] = "false"

VECTOR_DB = "vectordb"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Set your Groq API key in Streamlit secrets

# ---------------- DOMAIN ALIASES ----------------

DOMAIN_ALIAS = {
    "machine learning": "AI / Machine Learning",
    "ml": "AI / Machine Learning",
    "artificial intelligence": "AI / Artificial Intelligence",
    "ai": "AI / Artificial Intelligence",
    "cybersecurity": "Security / Cybersecurity",
    "cyber": "Security / Cybersecurity",
    "python": "Programming / Python",
    "networking": "Computer Science / Networking",
    "network": "Computer Science / Networking",
    "database": "Computer Science / Databases",
    "data structures": "Computer Science / Data Structures",
    "operating system": "Computer Science / OS",
}

# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Intelligent Library Assistant", page_icon="📚")

st.title("📚 Intelligent Library Assistant")

# ---------------- LOAD VECTOR DATABASE ----------------

@st.cache_resource
def load_db():

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=VECTOR_DB,
        embedding_function=embedding
    )

    return vectordb

vectordb = load_db()
retriever = vectordb.as_retriever(search_kwargs={"k":5})

# ---------------- LOAD LLM ----------------

@st.cache_resource
def load_llm():

    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

llm = load_llm()

# ---------------- METADATA EXTRACTION ----------------

def extract_metadata():

    data = vectordb._collection.get()

    titles=set()
    authors={}
    domains={}
    domain_counts={}

    for doc in data["documents"]:

        title=None
        author=None
        category=None

        for line in doc.split("\n"):

            if line.startswith("Title:"):
                title=line.replace("Title:","").strip()

            if line.startswith("Author:"):
                author=line.replace("Author:","").strip()

            if line.startswith("Category:"):
                category=line.replace("Category:","").strip()

        if title:

            titles.add(title)

            if author:
                authors[title]=author

            if category:

                domains[title]=category
                domain_counts[category]=domain_counts.get(category,0)+1

    return titles,authors,domains,domain_counts


book_titles,book_authors,book_domains,domain_counts=extract_metadata()

st.write(f"Total books in library: {len(book_titles)}")

# ---------------- SESSION MEMORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages=[]

if "last_books" not in st.session_state:
    st.session_state.last_books=[]

# ---------------- SEARCH FUNCTIONS ----------------

def search_by_title(query):

    q=query.lower()

    results=[]

    for title in book_titles:

        if q in title.lower():
            results.append(title)

    return results


def search_by_author(query):

    q=query.lower()

    results=[]

    for title,author in book_authors.items():

        if author and q in author.lower():
            results.append((title,author))

    return results


def detect_topic(query):

    q=query.lower()

    sorted_alias=sorted(DOMAIN_ALIAS.items(),key=lambda x:len(x[0]),reverse=True)

    for key,value in sorted_alias:

        pattern=r"\b"+re.escape(key)+r"\b"

        if re.search(pattern,q):
            return value

    return None


def most_popular_domain():

    if not domain_counts:
        return None

    return max(domain_counts,key=domain_counts.get)

# ---------------- LANGGRAPH STATE ----------------

class LibraryState(TypedDict):

    query:str
    context:str
    response:str
    docs:list
    messages:Annotated[list,add]

# ---------------- RETRIEVAL NODE ----------------

def retrieve_node(state:LibraryState):

    docs=retriever.invoke(state["query"])

    context=""

    for d in docs:
        context+=d.page_content+"\n"

    return {**state,"context":context,"docs":docs}

# ---------------- ANSWER NODE ----------------

def answer_node(state:LibraryState):

    if not state["context"]:

        return {
            **state,
            "response":"The library database does not contain this information."
        }

    prompt=f"""
Answer using ONLY the provided context.

Context:
{state["context"]}

Question:
{state["query"]}
"""

    response=llm.invoke(prompt)

    return {**state,"response":response.content}

# ---------------- BUILD GRAPH ----------------

def build_graph():

    graph=StateGraph(LibraryState)

    graph.add_node("retrieve",retrieve_node)
    graph.add_node("answer",answer_node)

    graph.add_edge(START,"retrieve")
    graph.add_edge("retrieve","answer")
    graph.add_edge("answer",END)

    return graph.compile()

workflow=build_graph()

# ---------------- CHAT HISTORY ----------------

for m in st.session_state.messages:

    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- USER INPUT ----------------

query=st.chat_input("Ask about library books")

if query:

    st.session_state.messages.append({"role":"user","content":query})

    with st.chat_message("user"):
        st.markdown(query)

    topic=detect_topic(query)

    answer=""

# ---------- TITLE SEARCH ----------

    title_results=search_by_title(query)

    if title_results:

        answer="### Matching Books\n\n"

        for t in title_results:

            author=book_authors.get(t,"Unknown")

            answer+=f"- {t} — {author}\n"

        st.session_state.last_books=title_results

# ---------- AUTHOR SEARCH ----------

    author_results=search_by_author(query)

    if author_results:

        answer="### Books by Author\n\n"

        for title,author in author_results:

            answer+=f"- {title} — {author}\n"

        st.session_state.last_books=[t for t,_ in author_results]

# ---------- TOPIC QUESTIONS ----------

    elif topic and "book" not in query.lower():

        books=[t for t,d in book_domains.items() if d==topic]

        if books:

            answer=f"You can learn about **{topic}** from these books:\n\n"

            for b in books:

                author=book_authors.get(b,"Unknown")

                answer+=f"- {b} — {author}\n"

            st.session_state.last_books=books

# ---------- STATISTICS ----------

    elif "how many books" in query.lower():

        answer=f"The library contains **{len(book_titles)} books**."

    elif "how many domains" in query.lower():

        answer=f"The library contains **{len(domain_counts)} domains**.\n\n"

        for d,c in domain_counts.items():
            answer+=f"- {d} → {c} books\n"

    elif "most popular domain" in query.lower():

        d=most_popular_domain()

        answer=f"The most popular domain is **{d}** with **{domain_counts[d]} books**."

# ---------- RECOMMENDATION ----------

    elif "recommend" in query.lower():

        recommended=list(book_titles)[:3]

        answer="### Recommended Books\n\n"

        for b in recommended:

            author=book_authors.get(b,"Unknown")

            answer+=f"- {b} — {author}\n"

# ---------- FOLLOW-UP MEMORY ----------

    elif "first one" in query.lower() and st.session_state.last_books:

        book=st.session_state.last_books[0]

        author=book_authors.get(book,"Unknown")

        answer=f"The first book is **{book}** written by **{author}**."

# ---------- RAG FALLBACK ----------

    else:

        initial_state={
            "query":query,
            "context":"",
            "response":"",
            "docs":[],
            "messages":[HumanMessage(content=query)]
        }

        result=workflow.invoke(initial_state)

        answer=result["response"]
        docs=result["docs"]

# ---------------- DISPLAY ----------------

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})

    if "docs" in locals() and docs:

        with st.expander("Retrieved Sources"):

            for i,d in enumerate(docs,1):

                st.markdown(f"Source {i}")

                st.write(d.page_content[:500])