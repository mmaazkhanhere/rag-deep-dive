import os
import sys
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Assuming retrieve_context_per_question is in helper_functions.retrieve_context_per_question
from helper_functions.retrieve_context_per_question import retrieve_context_per_question

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

file_path = "data/generative_ai_databricks.pdf"

def encode_file(path, chunk_size=1000, chunk_overlap=200):
    """
    This function will encode the PDF file into a vector store using OpenAI embeddings

    Arg:
    path: PDF book path
    chunk_size: size of the chunk
    chunk_overlap: overlap between chunks

    Returns:
    Vector store
    """
    logger.info(f"Starting to encode file: {path}")
    logger.info(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")

    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from the PDF.")
    except Exception as e:
        logger.error(f"Error loading PDF from {path}: {e}")
        raise

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    texts = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(texts)} raw text chunks.")

    cleaned_texts = [text for text in texts if text.page_content.strip()]
    logger.info(f"Cleaned texts resulted in {len(cleaned_texts)} non-empty chunks.")
    if not cleaned_texts:
        logger.warning("No clean text chunks found after splitting and stripping.")
        return None # Or handle as appropriate

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    logger.info("Initialized OpenAIEmbeddings.")

    try:
        vector_store = FAISS.from_documents(cleaned_texts, embeddings)
        logger.info("Successfully created FAISS vector store from documents.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}")
        raise


if __name__ == "__main__":
    logger.info("--- Starting script execution ---")

    try:
        chunks_vector_store = encode_file(file_path)
        if chunks_vector_store:
            logger.info("Vector store created successfully.")
            chunks_vector_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
            logger.info("Vector store retriever created with k=2.")

            query = "How should I prepare for Generative AI Certification?"
            logger.info(f"Performing context retrieval for query: \"{query}\"")

            context = retrieve_context_per_question(query, chunks_vector_retriever)
            logger.info(f"Retrieved {len(context)} contexts for the query.")
            
            for i, c in enumerate(context):
                print(f"Context {i + 1}:")
                print(c)
                print("\n")
        else:
            logger.error("Failed to create vector store. Exiting.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during script execution: {e}", exc_info=True)
    
    logger.info("--- Script execution finished ---")