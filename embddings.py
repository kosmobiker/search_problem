import os
import re
import html
import time
import openai
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "text-embedding-3-small"
BATCH_SIZE = 1024

def _clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return 'no data'
    text = html.unescape(text)
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    text = " ".join(text.split())
    return text if text else 'no data'

def get_openai_embeddings(texts):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            response = openai.embeddings.create(model=MODEL_NAME, input=batch)
            embeddings.extend([e.embedding for e in response.data])
            print(f"✅ Processed batch {i // BATCH_SIZE + 1}")
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(1)  # Pause before retrying
    return embeddings

def main():
    print("📥 Loading dataset...")
    df = pd.read_json('data/News_Category_Dataset_v3.json', lines=True, nrows=100)
    print(f"✅ Loaded dataset with {len(df)} records!")
    
    search_columns = ['headline', 'short_description', 'authors']
    for col in search_columns:
        df[col] = df[col].apply(_clean_text)
    
    print("📝 Combining text columns into 'documents'...")
    df["documents"] = df[search_columns].astype(str).agg(" ".join, axis=1)
    
    print("🤖 Creating embeddings, please wait...")
    df["embedding"] = get_openai_embeddings(df["documents"].tolist())
    
    output_file = "./data/test.parquet"
    df.to_parquet(output_file, compression="gzip")
    print(f"💾 Saved embeddings to {output_file}")

if __name__ == "__main__":
    main()