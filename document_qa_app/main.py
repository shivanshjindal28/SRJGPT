from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from typing import Optional
import google.generativeai as genai
from PIL import Image
from .qa.qa_engine import create_qa_chain, clear_vector_store
from .utils.text_extractor import extract_text_from_pdf, extract_text_from_docx
from .config import settings

# Configure Gemini
genai.configure(api_key=settings.google_api_key)
vision_model = genai.GenerativeModel('gemini-1.5-flash')
chat_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store the QA chain and last uploaded file info in memory
qa_chain = None
document_count = 0
last_file_type = None
last_file_path = None

class Question(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def process_image_question(image_path: str, question: str) -> str:
    try:
        # Load the image using PIL
        image = Image.open(image_path)
        
        # Generate response using Gemini Vision
        response = vision_model.generate_content([question, image])
        response.resolve()
        
        # Clean up the response text by removing double asterisks
        cleaned_text = response.text.replace('**', '')
        return cleaned_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

async def process_general_question(question: str) -> str:
    try:
        response = chat_model.generate_content(question)
        response.resolve()
        # Clean up the response text by removing double asterisks
        cleaned_text = response.text.replace('**', '')
        return cleaned_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global qa_chain, document_count, last_file_type, last_file_path
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Save the file
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Store file info
    last_file_path = file_path
    
    # Process based on file type
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        last_file_type = 'image'
        document_count += 1
        return {"message": f"Image uploaded successfully. Total documents: {document_count}"}
    elif file.filename.endswith('.pdf'):
        last_file_type = 'document'
        text = extract_text_from_pdf(file_path)
    elif file.filename.endswith('.docx'):
        last_file_type = 'document'
        text = extract_text_from_docx(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    if last_file_type == 'document':
        # Create QA chain for documents
        qa_chain = create_qa_chain(text, settings.google_api_key)
        document_count += 1
    
    return {"message": f"File uploaded and processed successfully. Total documents: {document_count}"}

@app.post("/ask")
async def ask_question(question: Question):
    global qa_chain, last_file_type, last_file_path
    
    try:
        if last_file_path:
            if last_file_type == 'image':
                # Process image-based question using Gemini
                answer = await process_image_question(last_file_path, question.question)
                return {
                    "answer": answer,
                    "sources": []  # No sources for image-based questions
                }
            else:
                # Process document-based question
                if qa_chain:
                    result = qa_chain.invoke({"query": question.question})
                    return {
                        "answer": result["result"],
                        "sources": [doc.page_content for doc in result["source_documents"]]
                    }
        
        # If no document is uploaded or for general questions
        answer = await process_general_question(question.question)
        return {
            "answer": answer,
            "sources": []  # No sources for general questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_documents():
    global qa_chain, document_count, last_file_type, last_file_path
    qa_chain = None
    document_count = 0
    last_file_type = None
    last_file_path = None
    clear_vector_store()
    return {"message": "All documents cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
