from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from document_qa_app.qa.qa_engine import create_qa_chain

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/api/query")
async def query(query: Query):
    try:
        # Get API key from environment variable
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=500,
                content={"message": "API key not configured"}
            )
        
        # Create QA chain (it will use existing vectors if available)
        chain = create_qa_chain("", api_key)
        
        # Get response
        response = chain.invoke({"query": query.question})
        
        return {
            "answer": response["result"],
            "sources": [doc.page_content for doc in response["source_documents"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 