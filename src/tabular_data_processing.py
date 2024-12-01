import pandas as pd
import numpy as np

def embed_tables_with_tapas(data, tapas_tokenizer, tapas_model):
    table_embeddings = []
    table_metadata = []

    for entry in data:
        for table_html in entry.get('tables', []):
            try:
                df = pd.read_html(str(table_html))[0]
                # Check if rows are empty before proceeding
                if len(df) <= 3:
                    continue  # Skip to the next table

                inputs = tapas_tokenizer(table=df, queries=["Summarize this table."], padding="max_length", truncation=True, return_tensors="pt")

                # Generate embeddings
                outputs = tapas_model(**inputs)
                embedding = outputs.pooler_output.detach().numpy()

                table_embeddings.append(embedding)
                table_metadata.append(table_html)
            except Exception as e:
                print(f"Error processing table: {e}")
    return np.vstack(table_embeddings), table_metadata



def flatten_df_rows(df):
    flattened_rows = "\n".join([", ".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.iterrows()])

    return flattened_rows


def embed_tables(data, text_model):
    table_embeddings = []
    table_metadata = []

    for entry in data:
        for table_html in entry.get('tables', []):
            try:
                df = pd.read_html(str(table_html))[0]
                # Check if rows are empty before proceeding
                if len(df) <= 3:
                    continue  # Skip to the next table

                flattened_df = flatten_df_rows(df)

                # Generate embeddings
                embedding = text_model.encode(flattened_df)

                table_embeddings.append(embedding)
                table_metadata.append(table_html)
            except Exception as e:
                print(f"Error processing table: {e}")
    return np.vstack(table_embeddings), table_metadata