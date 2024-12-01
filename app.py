# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:17:47 2024

@author: nkr4n
"""

import streamlit as st
import pandas as pd
from main import setup_rag_knowledge_base, retrieve_info, formulate_answer, display_top_pages
from src.data import load_data

logo = "data/logo.png"


# Display logo and title
_, col_logo, _ = st.columns([5,1,5])
col_logo.image(logo, width=100)
st.markdown("""<h2 style='text-align: center;'>Enterprise Knowledge Retrieval System</h2>""", unsafe_allow_html=True)


with st.sidebar:
    user_topic = st.sidebar.text_input(label="Setup Knowledge Base",placeholder="Enter your topic")
    num_docs = st.sidebar.slider("Choose no. of docs:")
    sidebar_submit = st.sidebar.button("Submit", use_container_width=True, type='secondary')
    if sidebar_submit:
        setup_rag_knowledge_base(user_topic, num_docs)

# Create a form for user input
col1,col2 = st.columns([4,1])
user_query = col1.text_input(label="Enter your query",placeholder="Enter your query", label_visibility='collapsed')
submitted = col2.button("Submit", use_container_width=True, type='primary')

if submitted:
    # Display user query
    text_results, table_results, image_results, text_model = retrieve_info(user_query, load_data("data/metadata.json"))
    raw_model_output = formulate_answer(user_query, text_results[:2])

    # Display table and image side-by-side
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_area(label="raw_model_output", value=raw_model_output, height=200, label_visibility='collapsed')
        st.dataframe(pd.read_html(table_results[0])[0], use_container_width=True, hide_index=True, height=200)
    with col2:
        st.image(image_results, use_container_width=True)

# Display page titles and links
    page_rank_dict = display_top_pages(user_query, load_data("data/wikipedia.json"), text_model)
    col1.write("### Relevant Pages")
    for title, url in page_rank_dict.items():
        col1.link_button(title, url, type="secondary", use_container_width=True)