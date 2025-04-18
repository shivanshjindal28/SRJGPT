from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from document_qa_app.qa.qa_engine import create_qa_chain, clear_vector_store
from PyPDF2 import PdfReader
from docx import Document
import io

app = FastAPI()

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = ""
        
        if file.filename.endswith('.pdf'):
            pdf = PdfReader(io.BytesIO(content))
            for page in pdf.pages:
                text += page.extract_text()
        elif file.filename.endswith('.docx'):
            doc = Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            return JSONResponse(
                status_code=400,
                content={"message": "Unsupported file format"}
            )
        
        # Get API key from environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"message": "API key not configured"}
            )
        
        # Create QA chain
        create_qa_chain(text, api_key)
        
        return {"message": "File processed successfully"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing file: {str(e)}"}
        ) 