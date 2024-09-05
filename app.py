from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
import logging
from firebase_setup import init_firebase
from pinecone_setup import init_pinecone
from utils import extract_text_from_pdf, validate_question, preprocess_text, get_embeddings, get_query_embeddings, generate_response
from pydantic import BaseModel
from dotenv import load_dotenv
from firebase_admin import firestore

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

# Load environment variables
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize services
db = init_firebase()
pinecone_index = init_pinecone()

app = FastAPI(title="RAG-Based Chat API")

class QueryRequest(BaseModel):
    chat_name: str
    question: str

@app.post("/upload_document")
async def upload_document(chat_name: str = Form(...), file: UploadFile = File(...)):
    temp_file_path = ""
    try:
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

        # Log the incoming file details
        logging.info(f"Received file: {file.filename} for chat: {chat_name}")

        file_id = str(uuid.uuid4())
        temp_file_path = f"temp_{file_id}.pdf"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Extract text from the PDF and log the status
        extracted_text = extract_text_from_pdf(temp_file_path)
        logging.info(f"Extracted text from PDF for chat: {chat_name}")

        # Preprocess the text and log the preprocessed result
        preprocessed_text = preprocess_text(extracted_text)
        logging.info(f"Preprocessed text for chat: {chat_name}: {preprocessed_text[:100]}...")  # Log a snippet

        if not preprocessed_text:
            raise HTTPException(status_code=400, detail="The uploaded PDF does not contain extractable text.")

        # Generate embeddings and log the embedding vector length
        embeddings = get_embeddings(preprocessed_text)
        logging.info(f"Generated embeddings for chat: {chat_name}, vector length: {len(embeddings)}")

        vector_id = f"{chat_name}_{file_id}"

        # Pinecone upsert and Firebase storage with error handling
        try:
            pinecone_index.upsert(vectors=[(vector_id, embeddings, {"chat_name": chat_name, "text": preprocessed_text})])
            logging.info(f"Upserted document into Pinecone for chat: {chat_name}, vector ID: {vector_id}")

            doc_ref = db.collection('documents').document(chat_name)
            doc_ref.set({
                'vector_id': vector_id,
                'file_name': file.filename,
                'uploaded_at': firestore.SERVER_TIMESTAMP
            })
            logging.info(f"Document metadata stored in Firebase for chat: {chat_name}")
        except Exception as e:
            logging.error(f"Error during Pinecone upsert or Firebase store: {str(e)}")
            pinecone_index.delete([vector_id])  # Rollback if Firebase fails
            raise HTTPException(status_code=500, detail="Failed to upload document.")
        
        os.remove(temp_file_path)
        return JSONResponse(status_code=200, content={"message": "Document uploaded and indexed successfully."})

    except HTTPException as he:
        logging.error(f"HTTPException occurred: {he.detail}")
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise he
    except Exception as e:
        logging.error(f"General exception occurred: {str(e)}")
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query_document")
def query_document(request: QueryRequest):
    """
    API 2: Document Querying
    """
    try:
        chat_name = request.chat_name
        question = request.question

        # Validate question
        if not validate_question(question):
            raise HTTPException(status_code=400, detail="Invalid or poorly formed question. Please ensure your question is clear and ends with a question mark.")

        # Retrieve the document index from Firebase
        doc_ref = db.collection('documents').document(chat_name)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Chat name not found.")

        vector_id = doc.to_dict().get('vector_id')
        if not vector_id:
            raise HTTPException(status_code=500, detail="Vector ID not found in metadata.")

        # Convert question to embeddings
        query_embeddings = get_query_embeddings(question)

        # Check if embeddings were successfully generated
        if not query_embeddings or not isinstance(query_embeddings, list):
            raise HTTPException(status_code=500, detail="Failed to generate query embeddings.")

        # Ensure vector dimensions match Pinecone index dimensions
        expected_dim = 384  # Adjust if your index has a different dimension
        if len(query_embeddings) != expected_dim:
            raise HTTPException(status_code=500, detail=f"Query embeddings dimension mismatch. Expected {expected_dim}, got {len(query_embeddings)}.")

        # Log query embeddings
        logging.debug(f"Query embeddings: {query_embeddings}")

        # Query Pinecone for relevant document sections
        response = pinecone_index.query(
            queries=[query_embeddings],
            top_k=5,
            include_metadata=True
        )

        # Validate Pinecone response
        if not response or 'results' not in response or not response['results']:
            raise HTTPException(status_code=404, detail="No relevant information found for the query.")

        # Aggregate retrieved content
        retrieved_content = ""
        for match in response['results'][0].get('matches', []):
            retrieved_content += match['metadata'].get('text', '') + "\n"

        if not retrieved_content.strip():
            raise HTTPException(status_code=404, detail="No relevant content retrieved.")

        # Generate response using Hugging Face model
        generated_response = generate_response(retrieved_content, question)

        # Ensure the generated response is valid
        if not generated_response:
            raise HTTPException(status_code=500, detail="Failed to generate response.")

        return JSONResponse(status_code=200, content={"response": generated_response})

    except HTTPException as he:
        raise he
    except Exception as e:
        # Log or print the exception for further debugging
        logging.error(f"Exception type: {type(e).__name__}, Message: {str(e)}")
    
        # Optional: Print traceback to see where the error originated from
        import traceback
        traceback.print_exc()
    
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/validate_question")
def validate_question_api(question: str = Form(...)):
    """
    API 3: Guardrails for Question Validation
    """
    is_valid = validate_question(question)
    if is_valid:
        return JSONResponse(status_code=200, content={"valid": True, "message": "Question is valid."})
    else:
        return JSONResponse(status_code=400, content={"valid": False, "message": "Invalid question. Please ensure your question is clear and appropriate."})
