from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import google.generativeai as genai
import numpy as np
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class Document:
    page_content: str
    metadata: dict = None

# Global variables
EMBEDDINGS_MODEL = None
DOCUMENTS = []
EMBEDDINGS = []

def compute_embeddings(texts: List[str]) -> np.ndarray:
    global EMBEDDINGS_MODEL
    if EMBEDDINGS_MODEL is None:
        EMBEDDINGS_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return EMBEDDINGS_MODEL.encode(texts, convert_to_tensor=False)

def similarity_search(query: str, k: int = 4) -> List[Document]:
    if not DOCUMENTS:
        return []
    
    query_embedding = compute_embeddings([query])[0]
    similarities = np.dot(EMBEDDINGS, query_embedding)
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    return [DOCUMENTS[i] for i in top_k_indices]

def create_qa_chain(text: str, google_api_key: str):
    global DOCUMENTS, EMBEDDINGS
    
    # Configure the Gemini API
    genai.configure(api_key=google_api_key)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    
    # Create documents and compute embeddings
    new_documents = [Document(page_content=chunk) for chunk in chunks]
    new_embeddings = compute_embeddings([doc.page_content for doc in new_documents])
    
    # Update global storage
    if not DOCUMENTS:
        DOCUMENTS.extend(new_documents)
        EMBEDDINGS = new_embeddings
    else:
        DOCUMENTS.extend(new_documents)
        EMBEDDINGS = np.vstack([EMBEDDINGS, new_embeddings])
    
    # Initialize Gemini Pro model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    
    class QAChain:
        def __init__(self, model):
            self.model = model
        
        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            try:
                question = inputs["query"]
                # Get relevant documents
                docs = similarity_search(question, k=4)
                context = "\n\n".join(doc.page_content for doc in docs)
                
                prompt = f"""Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
                
                generation_config = genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                )
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                if hasattr(response, 'text'):
                    answer = response.text
                else:
                    answer = str(response)
                
                return {
                    "result": answer,
                    "source_documents": docs
                }
            except Exception as e:
                print(f"Error in QAChain: {str(e)}")
                return {
                    "result": f"Error processing question: {str(e)}",
                    "source_documents": []
                }
    
    return QAChain(model)

def clear_vector_store():
    """Clear the stored documents and embeddings."""
    global DOCUMENTS, EMBEDDINGS
    DOCUMENTS = []
    EMBEDDINGS = np.array([])