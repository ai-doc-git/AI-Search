from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import faiss
import numpy as np


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return ' '.join(words)


def create_faiss_page_index(wikipedia_data, model):
    embeddings = []
    urls = []
    title = []
    for page in wikipedia_data:
        preprocessed_content = preprocess_text(page['content'])
        embedding = model.encode(preprocessed_content)
        embeddings.append(embedding)
        urls.append(page['url'])
        title.append(page['title'])

    embeddings = np.array(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, urls, title


