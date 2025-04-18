from typing import List, Dict, Any
import google.generativeai as genai
import numpy as np
from dataclasses import dataclass
import re
from collections import Counter

@dataclass
class Document:
    page_content: str
    metadata: dict = None

# Global variables
DOCUMENTS = []

def preprocess_text(text: str) -> List[str]:
    """Simple text preprocessing."""
    # Convert to lowercase and split into words
    words = re.findall(r'\w+', text.lower())
    # Remove common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
    return [w for w in words if w not in stop_words]

def similarity_search(query: str, k: int = 4) -> List[Document]:
    if not DOCUMENTS:
        return []
    
    # Preprocess query
    query_words = set(preprocess_text(query))
    
    # Calculate similarity scores using word overlap
    scores = []
    for doc in DOCUMENTS:
        doc_words = set(preprocess_text(doc.page_content))
        overlap = len(query_words & doc_words)
        scores.append(overlap)
    
    # Get top k documents
    top_k_indices = np.argsort(scores)[-k:][::-1]
    return [DOCUMENTS[i] for i in top_k_indices]

def create_qa_chain(text: str, google_api_key: str):
    global DOCUMENTS
    
    # Configure the Gemini API
    genai.configure(api_key=google_api_key)
    
    # Split text into chunks (simple splitting by sentences)
    chunks = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 50]
    
    # Create documents
    new_documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Update global storage
    if not DOCUMENTS:
        DOCUMENTS.extend(new_documents)
    else:
        DOCUMENTS.extend(new_documents)
    
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
    """Clear the stored documents."""
    global DOCUMENTS
    DOCUMENTS = []