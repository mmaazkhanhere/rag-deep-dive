import os
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.metrics.contextual_relevancy.contextual_relevancy import ContextualRelevancyMetric

from deepeval.test_case.llm_test_case import LLMTestCaseParams, LLMTestCase

from helper_functions.replace_t_with_space import replace_t_with_space

# Load environment variables
load_dotenv()


# Initialize LLM for answer generation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=os.getenv("GOOGLE_API_KEY"))

# Your original helper functions

# 1. Document Processing and Retrieval Setup
def setup_rag_components(path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFLoader(path)
    documents = loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)
    cleaned_docs = replace_t_with_space(split_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(cleaned_docs, embeddings)
    
    # Create BM25 index
    tokenized_docs = [doc.page_content.split() for doc in cleaned_docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    return vector_store, bm25, cleaned_docs

# 2. Retrieval Systems
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

def simple_retrieval(vector_store, query, k=5):
    return vector_store.similarity_search(query, k=k)

# 3. Answer Generation
def generate_answer(retriever, query, retrieval_func, bm25=None):
    if retrieval_func == fusion_retrieval:
        contexts = retrieval_func(vector_store, bm25, query)
    else:
        contexts = retrieval_func(vector_store, query)
    
    context_texts = [doc.page_content for doc in contexts]
    context_str = '\n'.join(context_texts)
    prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
    return llm.invoke(prompt), context_texts

# 4. Evaluation Setup
def create_test_cases(questions, rag_system, retrieval_func):
    test_cases = []
    for q in questions:
        answer, context = generate_answer(rag_system, q, retrieval_func, bm25)
        test_cases.append(LLMTestCase(
            input=q,
            actual_output=answer,
            retrieval_context=context,
            expected_output=questions[q]  # Ground truth answers
        ))
    return test_cases

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    model="gemini-2.5-flash",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
    threshold=0.7
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gemini-2.5-flash",
    include_reason=False
)

relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model="gemini-2.5-flash",
    include_reason=True
)

# Main Evaluation
if __name__ == "__main__":
    # Initialize RAG components
    path = "data/generative_ai_databricks.pdf"
    vector_store, bm25, cleaned_docs = setup_rag_components(path)
    
    # Define test questions and ground truth answers
    test_questions = {
        "What is Generative AI?": "Generative AI creates new content like text, images, or code.",
        "How does RAG work?": "RAG retrieves relevant documents and generates answers using context.",
        "What are the benefits of fusion retrieval?": "Combines keyword and semantic search for better relevance."
    }
    
    # Create test cases for both systems
    fusion_test_cases = create_test_cases(
        test_questions, 
        "Fusion RAG", 
        fusion_retrieval
    )
    
    simple_test_cases = create_test_cases(
        test_questions, 
        "Simple RAG", 
        simple_retrieval
    )
    
    # Evaluate Fusion RAG
    fusion_results = evaluate(
        fusion_test_cases,
        metrics=[correctness_metric, faithfulness_metric, relevancy_metric]
    )
    
    # Evaluate Simple RAG
    simple_results = evaluate(
        simple_test_cases,
        metrics=[correctness_metric, faithfulness_metric, relevancy_metric]
    )
    
    # Print comparison report
    print("\n" + "="*50)
    print("Fusion RAG Evaluation Results:")
    print(f"Correctness: {fusion_results.metric_scores.get('Correctness', 'N/A')}")
    print(f"Faithfulness: {fusion_results.metric_scores.get('Faithfulness', 'N/A')}")
    print(f"Contextual Relevancy: {fusion_results.metric_scores.get('Contextual Relevancy', 'N/A')}")
    
    print("\n" + "="*50)
    print("Simple RAG Evaluation Results:")
    print(f"Correctness: {simple_results.metric_scores.get('Correctness', 'N/A')}")
    print(f"Faithfulness: {simple_results.metric_scores.get('Faithfulness', 'N/A')}")
    print(f"Contextual Relevancy: {simple_results.metric_scores.get('Contextual Relevancy', 'N/A')}")
    
    # Additional metric details
    print("\n" + "="*50)
    print("Contextual Relevancy Reasons (Fusion RAG):")
    for case in fusion_test_cases:
        relevancy_metric.measure(case)
        print(f"Question: {case.input}")
        print(f"Reason: {relevancy_metric.reason}\n")