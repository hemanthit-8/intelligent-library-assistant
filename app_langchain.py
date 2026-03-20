import os
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

VECTOR_DB = "vectordb"

# Use environment variable for security
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ---------------- UI ----------------

st.set_page_config(page_title="Intelligent Library Assistant", page_icon="📚")

st.title("📚 Intelligent Library Assistant")

st.write("""
Ask questions about books in the library.

Examples:

• Best books for C programming  
• Python programming handbook  
• How many books are present  
• List book names  
• What domains exist  
• Books in python domain  
• How many books in AI
""")

# ---------------- LOAD VECTOR DB ----------------

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

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k":6}
)

# ---------------- LOAD LLM ----------------

@st.cache_resource
def load_llm():

    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        api_key=GROQ_API_KEY
    )


llm = load_llm()

# ---------------- EXTRACT BOOK METADATA ----------------

def extract_books():

    data = vectordb._collection.get()

    titles = set()
    domains = {}
    domain_counts = {}

    for doc in data["documents"]:

        title = None
        category = None

        for line in doc.split("\n"):

            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()

            if line.startswith("Category:"):
                category = line.replace("Category:", "").strip()

        if title:

            titles.add(title)

            if category:

                domains[title] = category

                if category not in domain_counts:
                    domain_counts[category] = 0

                domain_counts[category] += 1

    return titles, domains, domain_counts


book_titles, book_domains, domain_counts = extract_books()

st.write(f"**Total books in library:** {len(book_titles)}")

# ---------------- CHAT MEMORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- USER INPUT ----------------

query = st.chat_input("Ask about books in the library")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    q = query.lower()

    # ---------------- TOTAL BOOK COUNT ----------------

    if "how many books" in q and "in" not in q:

        answer = f"The library contains **{len(book_titles)} books**."

    # ---------------- LIST BOOKS ----------------

    elif "list books" in q or "name them" in q:

        answer = "### 📚 Books in Library\n\n"

        for b in sorted(book_titles):
            answer += f"- {b}\n"

    # ---------------- DOMAIN LIST ----------------

    elif "domain" in q or "categories" in q:

        answer = "### 📚 Domains in Library\n\n"

        for d, count in sorted(domain_counts.items()):
            answer += f"- **{d}** → {count} books\n"

        answer += f"\n**Total Domains:** {len(domain_counts)}"

    # ---------------- DOMAIN COUNT ----------------

    elif "how many books in" in q:

        domain_alias = {
            "ai": "AI / Artificial Intelligence",
            "artificial intelligence": "AI / Artificial Intelligence",
            "machine learning": "AI / Machine Learning",
            "ml": "AI / Machine Learning",
            "python": "Programming / Python",
            "c": "Programming / C",
            "cybersecurity": "Security / Cybersecurity",
            "network": "Computer Science / Networking",
            "networking": "Computer Science / Networking",
            "database": "Computer Science / Databases",
            "databases": "Computer Science / Databases",
            "os": "Computer Science / OS",
            "operating system": "Computer Science / OS"
        }

        matched_domain = None

        for key, value in domain_alias.items():

            if key in q:
                matched_domain = value
                break

        if matched_domain and matched_domain in domain_counts:

            count = domain_counts[matched_domain]

            answer = f"""
### 📚 Books in Domain

Domain: **{matched_domain}**

Total books available: **{count}**
"""

            answer += "\n### 📖 Book Titles\n"

            for title, domain in book_domains.items():
                if domain == matched_domain:
                    answer += f"- {title}\n"

        else:

            answer = "No books found for that domain."

    # ---------------- LIST BOOKS IN DOMAIN ----------------

    elif "books in" in q and "domain" in q:

        found = []

        for title, domain in book_domains.items():

            if domain.lower() in q:
                found.append(title)

        if found:

            answer = "### 📚 Books in Domain\n\n"

            for b in found:
                answer += f"- {b}\n"

        else:

            answer = "No books found in that domain."

    # ---------------- RAG SEARCH ----------------

    else:

        docs = retriever.invoke(query)

        if not docs:

            answer = "The library database does not contain information related to this query."

        else:

            context = ""

            for d in docs:
                context += d.page_content + "\n"

            prompt = f"""
You are an intelligent university library assistant.

Rules:
- Answer only using the provided context.
- Do not invent books or information.
- If information is missing say "The library database does not contain this information."

Context:
{context}

User Question:
{query}

Provide a clear and helpful explanation.
"""

            response = llm.invoke(prompt)

            answer = response.content

    # ---------------- DISPLAY ----------------

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # ---------------- SHOW SOURCES ----------------

    if "docs" in locals() and docs:

        with st.expander("📂 Retrieved Sources"):

            for i, d in enumerate(docs, 1):

                st.markdown(f"### Source {i}")

                st.write(d.page_content[:500])