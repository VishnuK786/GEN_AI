import os
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the blueprint
image_query_blueprint = Blueprint('image_query', __name__)

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="image_data",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Directory to persist the data
)

def get_top_documents_for_query(query, top_k=2):
    """Retrieve the top `top_k` documents most relevant to the given query."""
    # Generate the embedding for the query
    query_embedding = embeddings.embed_query(query)
    
    # Perform similarity search in the Chroma vector store
    top_documents = vector_store.similarity_search_by_vector(query_embedding, k=top_k)
    
    return top_documents

@image_query_blueprint.route('/highest-quality-image', methods=['GET'])
def highest_quality_image():
    query = request.args.get('query')
    top_k = int(request.args.get('top_k', 5))  # Default to 5 images if not provided

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    # Retrieve the top documents for the query
    top_documents = get_top_documents_for_query(query, top_k=top_k)

    if top_documents:
        images_data = []

        for i, document in enumerate(top_documents):
            image_url = document.metadata.get("image_url")  # Assuming metadata contains 'image_url'
            description = document.metadata.get("description", "No description available.")  # Optional description
            
            images_data.append({
                "image_url": image_url,
                "description": description,
            })
        
        return jsonify({"top_images": images_data})

    else:
        return jsonify({"error": "No relevant images found for the query"}), 404
