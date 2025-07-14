import os
from colorama import Fore, Back, Style
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI
from helper_functions.retrieve_context_per_question import retrieve_context_per_question
from helper_functions.replace_t_with_space import replace_t_with_space

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
    logger.info(f"{Fore.CYAN}Starting to encode file: {path}{Style.RESET_ALL}")
    logger.info(f"{Fore.CYAN}Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}{Style.RESET_ALL}")

    # 1. Prepare the knowledge base
    try:
        loader = PyPDFLoader(path) # load the PDF file
        documents = loader.load()
        logger.info(f"{Fore.GREEN}Loaded {len(documents)} pages from the PDF.{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Error loading PDF from {path}: {e}{Style.RESET_ALL}")
        raise
    
    # 2. Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # how many characters in each chunk
        chunk_overlap=chunk_overlap, # how many characters to overlap between chunks
        length_function=len # function to calculate the length of the text
    )

    # Split the documents into chunks
    texts = text_splitter.split_documents(documents)
    logger.info(f"{Fore.CYAN}Split documents into {len(texts)} raw text chunks.{Style.RESET_ALL}")

    cleaned_texts =replace_t_with_space(texts)  # Clean the text by replacing tabs with spaces
    logger.info(f"{Fore.CYAN}Cleaned texts resulted in {len(cleaned_texts)} non-empty chunks.{Style.RESET_ALL}")
    if not cleaned_texts:
        logger.warning(f"{Fore.YELLOW}No clean text chunks found after splitting and stripping.{Style.RESET_ALL}")
        return None # Or handle as appropriate

    # 3. Embed the chunks using HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    logger.info(f"{Fore.CYAN}Initialized OpenAIEmbeddings.{Style.RESET_ALL}")

    try:
        # 4. Create a FAISS vector store from the chunks
        vector_store = FAISS.from_documents(cleaned_texts, embeddings)
        logger.info(f"{Fore.GREEN}Successfully created FAISS vector store from documents.{Style.RESET_ALL}")
        return vector_store
    except Exception as e:
        logger.error(f"{Fore.RED}Error creating FAISS vector store: {e}{Style.RESET_ALL}")
        raise


if __name__ == "__main__":
    logger.info(f"{Fore.BLUE}--- Starting script execution ---{Style.RESET_ALL}")

    try:
        chunks_vector_store = encode_file(file_path)
        if chunks_vector_store:
            logger.info(f"{Fore.GREEN}Vector store created successfully.{Style.RESET_ALL}")
            chunks_vector_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})
            logger.info(f"{Fore.CYAN}Vector store retriever created with k=2.{Style.RESET_ALL}")

            
            # 5. Initialize the LLM 
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
            logger.info(f"{Fore.CYAN}Initialized ChatGoogleGenerativeAI model.{Style.RESET_ALL}")

            query = input("Enter your question: ")

            prompt_for_query_enhancing = """Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth. the document size has be exactly {chunk_size} characters."""

            enhanced_query = llm.invoke(prompt_for_query_enhancing.format(query=query, chunk_size=1000))

            logger.info(f"{Fore.CYAN}Enhanced query: {enhanced_query}{Style.RESET_ALL}")

            prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            logger.info(f"{Fore.CYAN}Pulled LangChain hub prompt.{Style.RESET_ALL}")

            # 6. Create the retrieval chain
            combine_docs_chain = create_stuff_documents_chain(
                llm, 
                prompt
            )
            logger.info(f"{Fore.CYAN}Created stuff documents chain.{Style.RESET_ALL}")

            qa_chain =  create_retrieval_chain(chunks_vector_retriever, combine_docs_chain)
            logger.info(f"{Fore.CYAN}Created retrieval chain.{Style.RESET_ALL}")
            
            # 7. Invoke the chain with the query
            response = qa_chain.invoke({"input": query})

            logger.info(f"{Fore.CYAN}Response from the retrieval chain:{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}{response['answer']}{Style.RESET_ALL}")
            # logger.info("Source documents:")
            # for doc in response['source_documents']:
            #     print(f"Document: {doc.metadata.get('source', 'Unknown Source')}")
            #     print(doc.page_content)
            #     print("\n")
        else:
            logger.error(f"{Fore.RED}Failed to create vector store. Exiting.{Style.RESET_ALL}")
    except Exception as e:
        logger.critical(f"{Fore.RED}An unhandled error occurred during script execution: {e}{Style.RESET_ALL}", exc_info=True)
    
    logger.info(f"{Fore.BLUE}--- Script execution finished ---{Style.RESET_ALL}")
