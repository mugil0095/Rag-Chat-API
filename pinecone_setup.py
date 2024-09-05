import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

def init_pinecone():
    # Fetch API key and other configurations from environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME").strip()  # Ensure no leading/trailing spaces

    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    if index_name not in pc.list_indexes().names():
        # Create index if it does not exist
        pc.create_index(
            name=index_name,
            dimension=384,  # Adjust according to your use case
            metric='cosine',  # or 'euclidean', etc.
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    # Return the Pinecone index object
    return pc.Index(index_name)
