
# RAG-Based Chat API

This API enables document uploading, querying, and intelligent response generation using a RAG-based (Retrieval Augmented Generation) framework. It integrates services like Firebase and Pinecone to store, index, and query documents. The text embeddings and responses are generated using Hugging Face models.

## Setup

### 1. Prerequisites
- Python 3.8+
- Firebase credentials JSON file (Firebase Admin SDK)
- Pinecone API key

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/rag-based-chat-api.git
cd rag-based-chat-api
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a `.env` file in the project root with the following configurations:
```plaintext
# Firebase configuration
FIREBASE_CREDENTIALS= "path/to/your/firebase/credentials.json"

# Pinecone API configuration
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_ENVIRONMENT="us-east-1-aws"
PINECONE_INDEX_NAME="rag-chat-index"

# Other configurations
DEBUG_MODE=True
```

### 5. Initialize Firebase
Ensure that your Firebase credentials JSON is correctly placed, and the `.env` file points to it.

### 6. Initialize Pinecone
Pinecone requires an API key and an index. The `init_pinecone()` function in `pinecone_setup.py` takes care of initializing the Pinecone client. It will create an index if it doesn't already exist.

### 7. Running the API
To run the FastAPI application locally, execute:
```bash
uvicorn app:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`.

### 8. API Endpoints

#### 1. **Upload Document**
**Endpoint**: `/upload_document`  
**Method**: `POST`  
**Parameters**:
- `chat_name`: The chat or session name (as `form-data`).
- `file`: PDF document (as `form-data`).

This API uploads a PDF document, extracts text, generates embeddings, and stores the embeddings in Pinecone and metadata in Firebase.

#### 2. **Query Document**
**Endpoint**: `/query_document`  
**Method**: `POST`  
**Body**:
```json
{
  "chat_name": "example_chat",
  "question": "What is the summary of the document?"
}
```

This API accepts a question and responds with relevant information from the document indexed under the given `chat_name`.

#### 3. **Validate Question**
**Endpoint**: `/validate_question`  
**Method**: `POST`  
**Parameters**:
- `question`: The question to validate.

This API checks if a question is well-formed (e.g., ends with a question mark).

### 9. Testing the APIs
You can use tools like [Postman](https://www.postman.com/) or [cURL](https://curl.se/) to test the API endpoints.

#### Example using `cURL`:
```bash
curl -X POST "http://127.0.0.1:8000/upload_document" -F "chat_name=test_chat" -F "file=@path/to/your/file.pdf"
```

#### Example using Postman:
- Use the POST method.
- Set the body to `form-data` and provide `chat_name` and `file`.

### 10. Additional Notes
- Ensure Pinecone is properly initialized with the correct index dimensions (e.g., 384) to match the embedding size.
- You can adjust the embedding and text generation models in `utils.py`.


