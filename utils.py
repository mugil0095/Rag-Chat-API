import PyPDF2
import re
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# For generating embeddings using Hugging Face transformers
def get_embeddings(text: str) -> list:
    """
    Generate embeddings for the given text using Hugging Face transformers.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling
    return embeddings

# Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Preprocess text for embedding and querying
def preprocess_text(text: str) -> str:
    """
    Preprocess the extracted text (e.g., remove special characters, normalize).
    """
    # Example: Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# For validating questions
def validate_question(question: str) -> bool:
    """
    Validate the incoming question to ensure it is well-formed and appropriate.
    """
    if not question or not question.strip():
        return False
    if not question.strip().endswith('?'):
        return False
    return True

# Generate embeddings for a question
def get_query_embeddings(question: str) -> list:
    """
    Generate embeddings for the given question using Hugging Face's embedding model.
    """
    return get_embeddings(question)

# Generate a response using Hugging Face GPT-style models
def generate_response(retrieved_content: str, question: str) -> str:
    """
    Generate a response using a Hugging Face GPT model.
    """
    generator = pipeline('text-generation', model='openai-community/gpt2')  # You can choose any transformer model
    prompt = f"Based on the following information:\n{retrieved_content}\nAnswer the question: {question}"
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']
