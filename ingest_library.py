import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_FOLDER = "loaded_pdfs"
VECTOR_DB = "vectordb"

documents = []

def extract_metadata(text):

    title = ""
    author = ""
    year = ""
    category = ""

    for line in text.split("\n"):

        if line.startswith("Title:"):
            title = line.replace("Title:", "").strip()

        if line.startswith("Author:"):
            author = line.replace("Author:", "").strip()

        if line.startswith("Published Year:"):
            year = line.replace("Published Year:", "").strip()

        if line.startswith("Category:"):
            category = line.replace("Category:", "").strip()

    return title, author, year, category


for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        path = os.path.join(PDF_FOLDER, file)

        loader = PyPDFLoader(path)

        pages = loader.load()

        for p in pages:

            title, author, year, category = extract_metadata(p.page_content)

            documents.append(
                Document(
                    page_content=p.page_content,
                    metadata={
                        "title": title,
                        "author": author,
                        "year": year,
                        "category": category
                    }
                )
            )


splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=VECTOR_DB
)

print("Vector DB created with", len(chunks), "chunks")

def main():

    # ✅ ADD THIS HERE (FIRST LINE inside main)
    if os.path.exists(VECTOR_DB):
        print("DB already exists")
        return

    print("Starting ingestion...")

    documents = []

    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):

            path = os.path.join(PDF_FOLDER, file)
            loader = PyPDFLoader(path)
            pages = loader.load()

            for p in pages:
                title, author, year, category = extract_metadata(p.page_content)

                documents.append(
                    Document(
                        page_content=p.page_content,
                        metadata={
                            "title": title,
                            "author": author,
                            "year": year,
                            "category": category
                        }
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=VECTOR_DB
    )

    print("Vector DB created with", len(chunks), "chunks")