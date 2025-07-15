import os
from colorama import Fore, Style
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate  # Added for custom prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def semantic_chunking_pdf(path, breakpoint_threshold=0.8):
    """
    Process PDF using semantic chunking and create vector store
    
    Args:
        path: Path to PDF file
        breakpoint_threshold: Similarity threshold for chunking (0.7-0.95)
    
    Returns:
        FAISS vector store with semantically chunked documents
    """
    logger.info(f"{Fore.CYAN}Starting semantic chunking for: {path}{Style.RESET_ALL}")
    
    try:
        # Load PDF document
        loader = PyPDFLoader(path)
        documents = loader.load()
        logger.info(f"{Fore.GREEN}Loaded {len(documents)} document pages{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}PDF loading error: {e}{Style.RESET_ALL}")
        raise

    # Initialize embeddings model (updated import path)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Create semantic chunker
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold
    )
    
    # Perform semantic chunking
    try:
        chunks = text_splitter.split_documents(documents)
        logger.info(f"{Fore.CYAN}Created {len(chunks)} semantic chunks{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Semantic chunking failed: {e}{Style.RESET_ALL}")
        raise

    # Create vector store
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info(f"{Fore.GREEN}Vector store created successfully{Style.RESET_ALL}")
        return vector_store
    except Exception as e:
        logger.error(f"{Fore.RED}Vector store creation error: {e}{Style.RESET_ALL}")
        raise

def get_answer(query, vector_store):
    """
    Get answer to a query using the vector store
    
    Args:
        query: User question
        vector_store: Created vector store
        
    Returns:
        Answer string and source documents
    """
    logger.info(f"{Fore.YELLOW}Processing query: {query}{Style.RESET_ALL}")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    # Create custom prompt that matches our variable names
    prompt_template = """
    Use the following context to answer the question at the end.
    If you don't know the answer, just say that you don't know.
    
    Context:
    {context}
    
    Question: {input}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Create retrieval chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Get response
    response = retrieval_chain.invoke({"input": query})
    return response["answer"], response["context"]

if __name__ == "__main__":
    logger.info(f"{Fore.BLUE}=== Starting Semantic QA System ==={Style.RESET_ALL}")
    
    try:
        # Process PDF with semantic chunking
        file_path = "data/generative_ai_databricks.pdf"
        vector_store = semantic_chunking_pdf(file_path, breakpoint_threshold=0.85)
        
        # Interactive question answering
        while True:
            print(f"\n{Fore.MAGENTA}Enter your question (type 'exit' to quit):{Style.RESET_ALL}")
            query = input(f"{Fore.CYAN}> {Style.RESET_ALL}")
            
            if query.lower() in ['exit', 'quit']:
                break
                
            if not query.strip():
                continue
                
            answer, context_docs = get_answer(query, vector_store)
            print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{answer}{Style.RESET_ALL}")
            
            # Show source information
            print(f"\n{Fore.YELLOW}Sources:{Style.RESET_ALL}")
            for i, doc in enumerate(context_docs, 1):
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'N/A')
                print(f"{i}. {os.path.basename(source)} (page {page+1})")
            
    except Exception as e:
        logger.critical(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
    
    logger.info(f"{Fore.BLUE}=== Session Ended ==={Style.RESET_ALL}")