import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from groq import Groq
from openai import OpenAI as OpenAIAPI
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
connection_string = "DefaultEndpointsProtocol=https;AccountName=genai123;AccountKey=EstDEzRJ+CP3SMlRlurcbK1cn8JAxoHT2VKfLwsZb+SZavm3p6uB8LtY2UqsTmskJYVK4BpdxmNm+AStjNtfIQ==;EndpointSuffix=core.windows.net"
container_name = "images-wvc"

client_llm = OpenAIAPI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="image_data",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Directory to persist the data
)

# Initialize Groq client
client = Groq()

# Initialize the LLM (language model)
llm = ChatGroq(
    model="llava-v1.5-7b-4096-preview",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)

def get_image_urls_from_blob(connection_string, container_name):
    """Fetch image URLs dynamically from Azure Blob storage."""
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    sas_expiry_time = datetime.utcnow() + timedelta(days=365)  # Set expiry for SAS token

    image_urls = []
    for blob in container_client.list_blobs():
        blob_name = blob.name
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=sas_expiry_time
        )
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        image_urls.append(blob_url)

    return image_urls

def get_image_description(image_url):
    """Fetch description for the image."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please list the following details based on the image, each in a separate paragraph:\n\n1. Full description of the image.\n2. Key features visible in the image (e.g., appearance, environment, context).\n3. Age of the person(s) in the image.\n4. Country they might be from.\n5. Sector or industry they may belong to (e.g., technology, healthcare, education, etc.)."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url  
                            }
                        }
                    ]
                }
            ],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in fetching description for {image_url}: {e}")
        return None

def generate_tags_from_description(description):
    """Generate tags based on the description."""
    prompt = f"""
    Based on the following description, generate relevant tags in the following sectors:
    
    **Featured Person**: Girl, Boy, Adolescent girl, Adolescent boy, Mother, Father, Community worker.
    **Sector**: Health, Education, Child Protection & Participation, WASH, Livelihoods.
    **Primary Topic Cluster**: Climate change, Disaster relief, Education, Food, Gender Equality, Health, International Development, Poverty, Refugees, Migrants and Displaced People, Social justice, Water.

    Description: {description}
    
    Please provide the tags under the respective categories as follows:
    - **Featured Person**: [tags]
    - **Sector**: [tags]
    - **Primary Topic Cluster**: [tags]
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client_llm.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for cheaper option
            messages=messages,
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generating tags: {e}")
        return None

def generate_embedding(description):
    """Generate an embedding vector for the description."""
    try:
        return embeddings.embed_query(description)
    except Exception as e:
        print(f"Error in generating embedding: {e}")
        return None

def save_to_chroma(image_data):
    """Save image data (descriptions and tags) to Chroma DB using Document format."""
    documents = []
    uuids = []
    embeddings_for_docs = []

    for data in image_data:
        description = data['description']
        tags = data['tags']
        image_url = data['image_url']
        
        # Combine description and tags into a single document for processing
        document_content = f"Description: {description}\nTags: {', '.join(tags)}"
        # print("Document content:")
        # print(document_content)
        # Create a Document with page_content as description and tags
        document = Document(
            page_content=document_content,
            metadata={"description": description, "tags": tags,"image_url" : image_url},
            id=str(uuid4())  # Generate a unique UUID for each document
        )
        # print("Document content 2:*********************************************")
        # print(document)
        
        # Add to the list of documents and uuids
        documents.append(document)
        uuids.append(document.id)
        
        # Generate embeddings for the document (combine description + tags)
        embeddings_for_docs.append(document.page_content)
    
    # Generate embeddings for the documents
    generated_embeddings = embeddings.embed_documents(embeddings_for_docs)

    # Save embeddings and metadata to Chroma DB
    vector_store.add_documents(
        documents=documents,
        embeddings=generated_embeddings,  # Pass the generated embeddings
        ids=uuids,
    )

    print(f"Successfully saved {len(image_data)} images to Chroma DB.")

def process_images(urls):
    """Process multiple images, extract data and store in Chroma DB."""
    image_data = []
    
    for url in urls:
        # Get the description for each image
        description = get_image_description(url)
        
        # Generate tags from description
        tags = generate_tags_from_description(description)
        
        # Collect image data
        image_info = {
            "description": description,
            "tags": tags,
            "image_url": url
        }
        image_data.append(image_info)
    
    # Save all the image data to Chroma DB
    print(image_data)
    save_to_chroma(image_data,)

# Main script
if __name__ == "__main__":
    try:
        # Fetch dynamic image URLs from Azure Blob storage
        urls = get_image_urls_from_blob(connection_string, container_name)

        # Process the images and store their data in Chroma DB
        process_images(urls)
    except Exception as e:
        print(f"Error: {e}")
