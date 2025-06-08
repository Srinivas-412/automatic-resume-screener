# my_utils.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_long_text(text: str):
    return model.encode(text)
