import os
from colorama import Fore, Back, Style
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from helper_functions.replace_t_with_space import replace_t_with_space


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


file_path = "data/generative_ai_databricks.pdf"

def encode_file(path, chunk_size=1000, chunk_overlap=200):
    logger.info(f"{Fore.CYAN}Starting to encode file: {path}{Style.RESET_ALL}")

    try:
        loader = PyPDFLoader(path)
        documents = loader.load()

        logger.info(f"{Fore.CYAN}Loaded {len(documents)} pages{Style.RESET_ALL}")

        # Splitting text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)

        texts = text_splitter.split_documents(documents)
        logger.info(f"{Fore.CYAN}Split documents into {len(texts)} chunks{Style.RESET_ALL}")

        # Cleaning text
        cleaned_text = replace_t_with_space(texts)
        logger.info(f"{Fore.CYAN}Cleaned into {len(cleaned_text)} chunks. {Style.RESET_ALL}")

        # Creating embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(cleaned_text, embeddings)
        logger.info(f"{Fore.CYAN}Vector store created{Style.RESET_ALL}")
        return vector_store
    except Exception as e:
        logger.error(f"{Fore.RED}Error encoding file: {e}{Style.RESET_ALL}")
        raise

if __name__ == "__main__":
    logger.info(f"{Fore.CYAN}Starting the main function{Style.RESET_ALL}")
    
    vector_store = encode_file(file_path)
    if not vector_store:
        logger.error(f"{Fore.RED}Vector store is empty{Style.RESET_ALL}")
        exit(1)

    # create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    logger.info(f"{Fore.CYAN}Retriever created{Style.RESET_ALL}")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, api_key = os.getenv("GOOGLE_API_KEY"))
    logger.info(f"{Fore.CYAN}LLM initialized{Style.RESET_ALL}")

    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    qa_chain = create_stuff_documents_chain(llm, prompt)
    logger.info(f"{Fore.CYAN}QA chain created{Style.RESET_ALL}")

    query = input("\nEnter your question: ")
    logger.info(f"{Fore.CYAN}User query: {query}{Style.RESET_ALL}")

    HYPO_PROMPT = """Generate a comprehensive document that answers this query: {query}
        - Create a detailed technical response
        - Include supporting explanations and examples
        - Maintain professional academic tone
        - Target length: ~1000 characters"""
    
    hypothetical_docs = llm.invoke(HYPO_PROMPT.format(query=query))
    logger.info(f"{Fore.CYAN}Hypothetical document generated{Style.RESET_ALL}")
    logger.info(f"{Fore.YELLOW}{hypothetical_docs.content}{Style.RESET_ALL}")

    context_docs = retriever.get_relevant_documents(hypothetical_docs.content)
    logger.info(f"{Fore.GREEN}Retrieved {len(context_docs)} context documents.{Style.RESET_ALL}")

    response = qa_chain.invoke({
            "input": query,
            "context": context_docs
        })

    print(f"\n{Back.BLUE}{Fore.WHITE} FINAL ANSWER {Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{response}{Style.RESET_ALL}")
    
    print(f"\n{Back.GREEN}{Fore.BLACK} SOURCE DOCUMENTS {Style.RESET_ALL}")
    for i, doc in enumerate(context_docs):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"{Fore.YELLOW}Source {i+1}: {source} (Page {page}){Style.RESET_ALL}")
        print(f"{doc.page_content[:250]}...\n")    

    logger.info(f"{Fore.BLUE}--- Pipeline Complete ---{Style.RESET_ALL}")

