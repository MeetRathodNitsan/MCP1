import os
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "testmcp"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)



def batch_iterate(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i: i + batch_size]



class EmbedData:
    def __init__(self, model_name="intfloat/multilingual-e5-large", batch_size=32):
        self.model_name = model_name
        self.embed_model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embeddings = []

    def embed(self, contexts):
        self.contexts = contexts
        self.embeddings = self.embed_model.encode(contexts, batch_size=self.batch_size, show_progress_bar=True)

    def get_query_embedding(self, query):
        return self.embed_model.encode(query)



class PineconeVDB:
    def __init__(self, index_name=INDEX_NAME, vector_dim=384):
        self.index_name = index_name
        self.vector_dim = vector_dim
        self._connect()

    def _connect(self):
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=self.vector_dim,
                metric="cosine"
            )
        self.index = pc.Index(self.index_name)

    def ingest_data(self, embeddata):
        for i, (text, vec) in enumerate(zip(embeddata.contexts, embeddata.embeddings)):
            metadata = {"text": text}
            self.index.upsert([(f"id-{i}", vec.tolist(), metadata)])



class Retriever:
    def __init__(self, index_name, embeddata):
        self.index = pc.Index(index_name)
        self.embeddata = embeddata

    def search(self, query, top_k=3):
        query_vector = self.embeddata.get_query_embedding(query)
        result = self.index.query(vector=query_vector.tolist(), top_k=top_k, include_metadata=True)

        hits = result.get("matches", [])
        combined_prompt = []
        for hit in hits:
            combined_prompt.append(hit["metadata"]["text"])

        return "\n\n---\n\n".join(combined_prompt)
