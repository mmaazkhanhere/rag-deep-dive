import os
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from helper_functions.replace_t_with_space import replace_t_with_space
from helper_functions.show_context import show_context

load_dotenv()

path = "data/generative_ai_databricks.pdf"

def encode_pdf_and_get_split_docs(path, chunk_size=1000, chunk_overlap=200):
    
    loader = PyPDFLoader(path)
    documents = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    text = text_splitter.split_documents(documents)

    cleaned_text = replace_t_with_space(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = FAISS.from_documents(cleaned_text, embeddings)
    return vector_store, cleaned_text


def create_bm25_index(documents: list[Document]):
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)


def fusion_retrieval(vector_store, bm25, query: str, k: int = 5, alpha: float = 0.5)->list[Document]:
    epsilon = 1e-8

    # 1. Get all documents from vector store
    all_docs = vector_store.similarity_search("", k=vector_store.index.ntotal)
    
    # 2. Perform BM25 search
    bm25_scores = bm25.get_scores(query.split())

    # 3. Perform vector search
    vector_results = vector_store.similarity_search_with_score(query, k=len(all_docs))

    # 4. Normalize scores

    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) -  np.min(bm25_scores) + epsilon)

    # 5. Combine scores
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores  

    # 6. Rank documents
    sorted_indices = np.argsort(combined_scores)[::-1]

    # 7. return top k documents
    return [all_docs[i] for i in sorted_indices[:k]]


vector_store, cleaned_text = encode_pdf_and_get_split_docs(path)
bm25 = create_bm25_index(cleaned_text) 

query = input("Ask question: ")
top_docs = fusion_retrieval(vector_store, bm25, query, k=5, alpha=0.5)
docs_content = [doc.page_content for doc in top_docs]
show_context(docs_content)
