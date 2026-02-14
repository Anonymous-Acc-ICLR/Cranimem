# File: cranimem/core/embedding.py
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from..config import settings

def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
