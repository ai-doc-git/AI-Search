from io import BytesIO
import requests
import numpy as np
from PIL import Image


def embed_images_with_blip(data, blip_processor, blip_model, text_model):
    image_embeddings = []
    image_metadata = []

    for entry in data:
        for url in entry.get('images', []):
            try:
                # Download image
                response = requests.get(url, stream=True)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert("RGB")

                # Generate caption using BLIP
                inputs = blip_processor(images=img, return_tensors="pt")
                outputs = blip_model.generate(**inputs)
                caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
                
                print(f"For Image: {url}, the generated caption is {caption}")

                # Generate text embedding from caption
                embedding = text_model.encode([caption])
                image_embeddings.append(embedding)
                image_metadata.append(url)

            except Exception as e:
                print(f"Error processing image {url}: {e}")

    return np.vstack(image_embeddings), image_metadata


