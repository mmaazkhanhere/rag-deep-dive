import os
import faiss
from colorama import Fore, Back, Style
import logging
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


if __name__ == "__main__":
    logger.info(f"{Fore.BLUE}--- Starting script execution ---{Style.RESET_ALL}")

    file_path = "data/marketing_campaign.csv"
    data = pd.read_csv(file_path)
    logger.info(f"{Fore.BLUE}Data loaded from {file_path}", Style.RESET_ALL)

    loader = CSVLoader(file_path)
    docs = loader.load()
    logger.info(f"{Fore.BLUE}Documents loaded from CSV file{Style.RESET_ALL}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    vector_store.add_documents(documents=docs)

    logger.info(f"{Fore.BLUE}Documents added to FAISS vector store{Style.RESET_ALL}")

    retriever = vector_store.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    query = input("Enter your question: ")
    answer = rag_chain.invoke({"input": query})
    logger.info(f"{Fore.GREEN}Answer: {answer['answer']}{Style.RESET_ALL}")


    logger.info(f"{Fore.BLUE}--- Script execution finished ---{Style.RESET_ALL}")
