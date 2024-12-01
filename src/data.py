import wikipediaapi
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from PIL import Image

def fetch_wikipedia_page(title, wiki_obj):
    page = wiki_obj.page(title)
    
    if not page.exists():
        return None

    # Text
    content = page.text

    # URL
    url = page.fullurl
    response = requests.get(url, headers={"User-Agent": "EducationalScript/1.0 (nkr4nikhilraj@gmail.com)"})
    
    if response.status_code != 200:
        print(f"Failed to fetch page content for {title}.")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')

    # Tables
    tables = [str(table) for table in soup.find_all('table')]

    # Images
    images = ["https:" + img['src'] for img in soup.find_all('img') if 'src' in img.attrs]
    valid_images = filter_valid_images(images, min_size=200)

    return {
        'title': page.title,
        'url': page.fullurl,
        'content': content,
        'tables': tables,
        'images': valid_images,
    }



def search_and_fetch_wikipedia(topic, wiki_obj, max_pages=50):
    
    search_url = f"https://en.wikipedia.org/w/index.php?search={topic}&title=Special:Search&limit={max_pages}&profile=advanced&fulltext=1&ns0=1"
    response = requests.get(search_url, headers={"User-Agent": "EducationalScript/1.0 (nkr4nikhilraj@gmail.com)"})
    
    if response.status_code != 200:
        print("Failed to fetch search results.")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    divs = soup.find_all('div', class_='mw-search-result-heading')

    links = [
        f"https://en.wikipedia.org{a['href']}"
        for div in divs
        for a in div.find_all('a', href=True)
]

    data = []
    for link in links[:max_pages]:
        title = link.split('/')[-1]  # Extract title from URL
        page_data = fetch_wikipedia_page(title, wiki_obj)
        if page_data:
            data.append(page_data)

    return data



def filter_valid_images(image_urls, min_size=200):
    valid_images = []
    for url in image_urls:
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Check image size
            image = Image.open(response.raw)
            if image.width >= min_size and image.height >= min_size:
                valid_images.append(url)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image: {url} - {e}")
        except OSError as e:
            print(f"Error opening image: {url} - {e}")

    return valid_images


def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)