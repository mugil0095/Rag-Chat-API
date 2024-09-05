import os
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

def init_firebase():
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve Firebase credentials path from environment variable
    cred_path = os.getenv("FIREBASE_CREDENTIALS")
    if not cred_path:
        raise ValueError("Firebase credentials path missing in .env file")
    
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    return db
