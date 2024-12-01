from src.data import load_data, search_and_fetch_wikipedia
from src.text_data_processing import *
from src.tabular_data_processing import *
from src.image_data_processing import *
from src.page_ranking import *
from src.gen_ai import *
from src.utilities import *

import wikipediaapi
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def setup_rag_knowledge_base(topic, max_pages):
    
    # Step 1: Get the data as per user specified topic and max pages
    data = search_and_fetch_wikipedia(topic=topic, 
                                  wiki_obj=wikipediaapi.Wikipedia(user_agent="I am using it for educational purpose.", language='en'), 
                                  max_pages=max_pages)
    
    # Step 2: Dump the data locally in json format
    with open('data/wikipedia.json', 'w') as file:
        json.dump(data, file, indent=4)
        
    # Step 3: Load the models for text, table and image processing    
    text_model, tapas_tokenizer, tapas_model, blip_processor, blip_model = initialize_models('all')

    text_embeddings, text_metadata = embed_text(data, text_model, chunk_size=5, overlap_size=0)
    table_embeddings, table_metadata = embed_tables(data, text_model)
    image_embeddings, image_metadata = embed_images_with_blip(data, blip_processor, blip_model, text_model)
    
    metadata_dict = {}
    metadata_dict['text_data'] = text_metadata
    metadata_dict['tabular_data'] = table_metadata
    metadata_dict['image_data'] = image_metadata
    
    with open("data/metadata.json", "w") as outfile: 
        json.dump(metadata_dict, outfile)
    
    
    text_index = build_and_save_faiss_index(text_embeddings, "vector_store/text_index.faiss")
    table_index = build_and_save_faiss_index(table_embeddings, "vector_store/table_index.faiss")
    image_index = build_and_save_faiss_index(image_embeddings, "vector_store/image_index.faiss")
    

def retrieve_info(user_query, metadata):
    
    text_model = initialize_models(type='text')
    
    text_index = load_faiss_index("vector_store/text_index.faiss")
    table_index = load_faiss_index("vector_store/table_index.faiss")
    image_index = load_faiss_index("vector_store/image_index.faiss")

    text_metadata = metadata['text_data']
    table_metadata = metadata['tabular_data']
    image_metadata = metadata['image_data']

    user_query_embedding = text_model.encode([user_query], convert_to_tensor=False)
    
    text_distances, text_indices = retrieve_top_k(user_query_embedding, text_index, k=1)
    text_results = [text_metadata[i] for i in text_indices[0]]
    
    table_distances, table_indices = retrieve_top_k(user_query_embedding, table_index, k=5)
    table_results = [table_metadata[i] for i in table_indices[0]]
    
    image_distances, image_indices = retrieve_top_k(user_query_embedding, image_index, k=1)
    image_results = [image_metadata[i] for i in image_indices[0]]
    
    return text_results, table_results, image_results, text_model


def formulate_answer(user_query, retrieved_context):
    raw_model_output = generate_answer(user_query, retrieved_context)
    model_output = raw_model_output.split(user_query)[1].strip()
    
    return model_output


def display_top_pages(user_query, wikipedia_data, text_model):
    
    index, urls, title = create_faiss_page_index(wikipedia_data, text_model)
    query_embedding = text_model.encode(user_query)
    distances, indices = index.search(query_embedding.reshape(1, -1), k=5)
    
    page_data = [(title[i], urls[i]) for i in indices.flatten()]
    sorted_pages = sorted(zip(distances.flatten(), page_data))
    page_rank_dict = {title: url for distance, (title, url) in sorted_pages}
    
    return page_rank_dict



if __name__ == "__main__":
    pass
    # setup_rag_knowledge_base("Google", 25)
#     user_query = "What is Google Maps?"
#     text_results, table_results, image_results, text_model = retrieve_info(user_query, load_data("data/metadata.json"))
#     raw_model_output = formulate_answer(user_query, " ".join(text_results[:2]))
    
#     print(raw_model_output)
    
#     page_rank_dict = display_top_pages(user_query, load_data("data/wikipedia.json"), text_model)
#     print(page_rank_dict)
    
    