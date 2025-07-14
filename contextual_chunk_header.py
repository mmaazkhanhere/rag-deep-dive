import os
from colorama import Fore, Back, Style
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pdf_path = "data/generative_ai_databricks.pdf"

# Updated Pydantic model with proper list type declaration
class DocumentMetaData(BaseModel):
    document_title: str = Field(description="Title of the document")
    primary_topic: str = Field(description="Primary topic of the document")
    key_entities: list[str] = Field(description="List of key entities in the document")

def enhanced_metadata_extraction(text: str):
    """Extract metadata using Gemini with structured output"""
    llm = ChatGoogleGenerativeAI(
        temperature=0, 
        model="gemini-2.5-flash",  # Updated to latest flash model
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    structure_llm = llm.with_structured_output(DocumentMetaData)
    return structure_llm.invoke(text)

def create_contextual_chunks(pdf_path: str):
    """Main function to load PDF and create contextual chunks"""
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 2. Extract metadata from first page
    first_page_text = documents[0].page_content
    try:
        metadata = enhanced_metadata_extraction(first_page_text)
        logger.info(f"Extracted metadata: {metadata}")
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        # Fallback metadata
        metadata = DocumentMetaData(
            document_title=os.path.basename(pdf_path),
            primary_topic="Generative AI",
            key_entities=[]
        )
    
    # 3. Create chunks with contextual headers
    section_chunks = []
    for i, doc in enumerate(documents):
        # Create section-aware header
        section_header = (
            f"CONTEXTUAL HEADER || "
            f"Document: {metadata.document_title} | "
            f"Topic: {metadata.primary_topic} | "
            f"Page: {i+1}/{len(documents)}\n\n"
        )
        
        # Split page content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        page_chunks = splitter.split_text(doc.page_content)
        
        # Add header to each chunk
        section_chunks.extend([section_header + chunk for chunk in page_chunks])
    
    logger.info(f"Created {len(section_chunks)} contextual chunks")
    return section_chunks

# Execute the processing
if __name__ == "__main__":
    print(f"{Fore.GREEN}Processing PDF: {pdf_path}{Style.RESET_ALL}")
    chunks = create_contextual_chunks(pdf_path)
    
    # Display sample chunk
    print(f"\n{Fore.CYAN}Sample Contextual Chunk:{Style.RESET_ALL}")
    print(chunks[0][:500] + "...")