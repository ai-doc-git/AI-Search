import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import TapasTokenizer, TapasModel, BlipProcessor, BlipForConditionalGeneration, pipeline


def initialize_models(type='text'):
    if type == 'text':
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        return text_model
    else:
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        tapas_tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
        tapas_model = TapasModel.from_pretrained("google/tapas-base-finetuned-wtq")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return text_model, tapas_tokenizer, tapas_model, blip_processor, blip_model



def build_and_save_faiss_index(embeddings, index_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index


# Load FAISS index
def load_faiss_index(index_path):
    return faiss.read_index(index_path)



def pad_embedding(embedding, target_dimension):
    embedding_dimension = embedding.shape[1]

    if embedding_dimension < target_dimension:
        # Pad with zeros
        padding_shape = (embedding.shape[0], target_dimension - embedding_dimension)  # Create a tuple
        padding = np.zeros(padding_shape, dtype=embedding.dtype)
        padded_embedding = np.concatenate([embedding, padding], axis=1)
        return padded_embedding
    elif embedding_dimension > target_dimension:
        # Truncate
        return embedding[:, :target_dimension]
    else:
        # Dimensions match, no change needed
        return embedding
    
    
    
# Retrieve top-k results
def retrieve_top_k(query_embedding, index, k=5):
    # Check if query embedding dimension matches index dimension
    query_embedding_dimension = query_embedding.shape[1]  # Get column dimension
    index_dimension = index.d  # Assumed correct attribute name

    if query_embedding_dimension != index_dimension:
        # Adjust the embedding to match the index dimension
        print(f"Query embedding dimension ({query_embedding_dimension}) does not match index dimension ({index_dimension}). Hence, embeddings are adjusted.")

        query_embedding = pad_embedding(query_embedding, index_dimension)

    # Now perform the search
    distances, indices = index.search(query_embedding, k)
    return distances, indices