Step-by-step Instructions:

1. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  (or venv\Scripts\activate on Windows)

2. Install dependencies:
   pip install -r requirements.txt

3. Set your OpenAI API key (export in terminal or load from .env file):
   export OPENAI_API_KEY=your_key_here

4. Run the server:
   uvicorn document_qa_app.main:app --reload

5. Server will be available at:
   http://127.0.0.1:8000

6. Use /upload endpoint to test with file and question.