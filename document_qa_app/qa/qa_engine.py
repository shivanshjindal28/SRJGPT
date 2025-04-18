from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from typing import List, Dict, Any

# Global vector store and embeddings
VECTOR_STORE = None
EMBEDDINGS = None

def create_qa_chain(text: str, google_api_key: str):
    global VECTOR_STORE, EMBEDDINGS
    
    # Configure the Gemini API
    genai.configure(api_key=google_api_key)
    
    # Initialize embeddings if not already initialized
    if EMBEDDINGS is None:
        EMBEDDINGS = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    
    # Create or update vector store
    if VECTOR_STORE is None:
        VECTOR_STORE = Chroma.from_texts(chunks, EMBEDDINGS)
    else:
        # Add new chunks to existing vector store
        VECTOR_STORE.add_texts(chunks)
    
    # Initialize Gemini Pro model with the correct model name
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    
    class QAChain:
        def __init__(self, model, vector_store):
            self.model = model
            self.vector_store = vector_store
        
        def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            try:
                question = inputs["query"]
                # Get relevant documents
                docs = self.vector_store.similarity_search(question, k=4)
                context = "\n\n".join(doc.page_content for doc in docs)
                
                # Create the prompt
                prompt = f"""Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
                
                # Generate response with simple configuration
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
    
    return QAChain(model, VECTOR_STORE)

def clear_vector_store():
    """Clear the vector store."""
    global VECTOR_STORE
    VECTOR_STORE = None