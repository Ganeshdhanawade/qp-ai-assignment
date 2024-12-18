import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections

class VectorStore:
    def __init__(self, milvus_host="localhost", milvus_port="19530", collection_name="document_chunks"):
        # Initialize Milvus connection
        connections.connect(host=milvus_host, port=milvus_port)
        
        # Milvus collection schema setup (Embedding vectors + chunks as strings)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = collection_name

        # Define the schema for the collection
        self.fields = [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # embedding dimension
            FieldSchema(name="chunk", dtype=DataType.STRING, is_primary=True)  # chunk of text
        ]
        
        # Check if the collection exists, if not, create it
        if not self._check_if_collection_exists():
            self._create_collection()

        self.collection = Collection(name=self.collection_name)

        self.embeddings = []
        self.chunks = []

    def _check_if_collection_exists(self):
        """Check if the collection already exists in Milvus."""
        collections = Collection.list()
        return self.collection_name in collections

    def _create_collection(self):
        """Create a Milvus collection."""
        schema = CollectionSchema(fields=self.fields, description="Document chunks with embeddings")
        Collection(name=self.collection_name, schema=schema)

    def add_chunks(self, chunks):
        """Add document chunks to Milvus vector store."""
        embeddings = self.embedder.encode(chunks)
        
        # Insert embeddings and chunks into Milvus
        data = [
            embeddings.tolist(),  # List of embeddings
            chunks  # List of chunks (texts)
        ]
        self.collection.insert(data)

        # Store embeddings and chunks in memory
        self.embeddings.extend(embeddings)
        self.chunks.extend(chunks)

    def get_relevant_chunks(self, query, top_k=3):
        """Retrieve top_k most relevant chunks based on semantic similarity."""
        query_embedding = self.embedder.encode([query])
        
        # Search the collection in Milvus for the top_k most similar vectors
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            query_embedding.tolist(),  # Query embedding (list of list)
            "embedding",  # The field to search
            search_params,
            limit=top_k  # Top K similar results
        )

        # Extract the chunk data from the search results
        relevant_chunks = [result.entity.get('chunk') for result in results[0]]
        
        return relevant_chunks


if __name__ == "__main__":
    # Example usage:
    vector_store = VectorStore()

    # Add chunks to the store
    chunks = [
        "This is the first chunk of the document.",
        "This is the second chunk of the document.",
        "This is the third chunk of the document."
    ]
    vector_store.add_chunks(chunks)

    # Query and get relevant chunks
    query = "Tell me about the document"
    relevant_chunks = vector_store.get_relevant_chunks(query)

    print("Relevant chunks:", relevant_chunks)