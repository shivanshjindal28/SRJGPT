from typing import List, Dict, Any
import google.generativeai as genai
import numpy as np
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Document:
    page_content: str
    metadata: dict = None

# Global variables
VECTORIZER = None
DOCUMENTS = []
VECTORS = None

def compute_vectors(texts: List[str]) -> np.ndarray:
    global VECTORIZER
    if VECTORIZER is None:
        VECTORIZER = TfidfVectorizer(stop_words='english')
        return VECTORIZER.fit_transform(texts)
    return VECTORIZER.transform(texts)

def similarity_search(query: str, k: int = 4) -> List[Document]:
    if not DOCUMENTS:
        return []
    
    query_vector = VECTORIZER.transform([query])
    similarities = cosine_similarity(query_vector, VECTORS).flatten()
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    return [DOCUMENTS[i] for i in top_k_indices]

def create_qa_chain(text: str, google_api_key: str):
    global DOCUMENTS, VECTORS
    
    # Configure the Gemini API
    genai.configure(api_key=google_api_key)
    
    # Split text into chunks (simple splitting by sentences)
    chunks = [s.strip() for s in text.split('.') if len(s.strip()) > 50]
    
    # Create documents and compute vectors
    new_documents = [Document(page_content=chunk) for chunk in chunks]
    new_vectors = compute_vectors([doc.page_content for doc in new_documents])
    
    # Update global storage
    if not DOCUMENTS:
        DOCUMENTS.extend(new_documents)
        VECTORS = new_vectors
    else:
        DOCUMENTS.extend(new_documents)
        VECTORS = np.vstack([VECTORS.toarray(), new_vectors.toarray()])
        VECTORS = VECTORS.reshape(-1, VECTORS.shape[1])
    
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
    """Clear the stored documents and vectors."""
    global DOCUMENTS, VECTORS, VECTORIZER
    DOCUMENTS = []
    VECTORS = None
    VECTORIZER = None