import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

def chunk_text_by_sentences(text, chunk_size=5, overlap_size=2):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), chunk_size - overlap_size):
        chunk = sentences[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks



def embed_text(data, text_model, chunk_size=5, overlap_size=2):
    all_chunks = []
    for entry in data:
        chunks = chunk_text_by_sentences(entry['content'], chunk_size, overlap_size)
        all_chunks.extend(chunks)

    # Generate aaaembeddings for all chunks
    text_embeddings = text_model.encode(all_chunks, convert_to_tensor=False)
    return np.array(text_embeddings), all_chunks



