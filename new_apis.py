import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, OpenAI
from crewai_tools import tool, FileWriterTool, FileReadTool
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from crewai import LLM
from flask_cors import CORS
from os import scandir
from image_query import image_query_blueprint 
from langtrace_python_sdk import langtrace
import requests
import json
import shutil
from azure.storage.blob import BlobServiceClient
import fitz 
import pandas as pd
import ast
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from chroma_db_data_creation import process_folder

# Must precede any llm module imports
langtrace.init(api_key = os.getenv("LANGTRACE_API_KEY"))

app = Flask(__name__)
app.register_blueprint(image_query_blueprint, url_prefix='/image-query')

linkedin_access_token = os.getenv("LINKEDIN_ACCESS_TOKEN")

CORS(app)

# Load environment variables
load_dotenv()

# Global variable to store ChromaDB instance
chroma_db_instance = None

# API Type Configuration
api_type = 2  # 1 - Azure OpenAI, 2 - OpenAI

if api_type == 1:
    azure_endpoint = os.getenv("Azure_API_ENDPOINT")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = "2023-03-15-preview"
    deployment_name = "gpt-4o"
    embeddings = AzureOpenAIEmbeddings(azure_endpoint=azure_endpoint, api_key=api_key)
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        model=deployment_name,
        temperature=1,
        max_tokens=300,
    )
else:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(max_tokens=300)


# Chroma initialization function
def init_chroma_db(folder_name: str, file_name: str):
    global chroma_db_instance
    # Define the persistent directory where ChromaDB data will be stored
    persistent_directory = os.path.join(folder_name, file_name)
    
    if chroma_db_instance is None:
        chroma_db_instance = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )
        print(f"Initialized ChromaDB at {persistent_directory}")
    else:
        print(f"Using existing ChromaDB instance at {persistent_directory}")
    
    return chroma_db_instance

# Custom tool for querying ChromaDB
@tool
def query_chroma_db(query_text: str, folder_name: str, file_name: str) -> str:
    """
    Retrieves relevant chunks from Chroma DB using a semantic search
    based on the input query_text.
    """
    # Initialize ChromaDB (or reuse existing)
    results = chroma_db_instance.similarity_search(query_text, k=10)
    formatted_results = "\n\n".join([doc.page_content for doc in results if hasattr(doc, 'page_content')])
    return formatted_results

# Helper function: Generate output file path
def generate_output_file_path(collection_name: str, file_type: str) -> str:
    return os.path.join("outputs", f"{file_type}_{collection_name}.txt")

# Helper function: Read file content
def read_file_content(file_path: str) -> str:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None

# Define Agents
# Data Retrieval Agent
data_retriever = Agent(
    role="Data Researcher",
    goal="Retrieve relevant information about the blog topic from the Chroma vector database.",
    verbose=True,
    memory=True,
    backstory="An AI specialized in locating relevant research and information from a vast Chroma DB.",
    tools=[query_chroma_db]
)

# Blog Writing Agent
blog_writer = Agent(
    role="Blog Writer",
    goal="Write an informative blog post using the retrieved data from Chroma DB and referencing sample_blog.txt.",
    verbose=True,
    memory=True,
    backstory="An AI writer proficient in crafting insightful blog posts that clarify complex topics.",
    # New tools with file paths provided to FileReadTool
    tools=[FileReadTool(file_path='samples/sample_blog.txt'), FileWriterTool()]  
)

# Instagram Conversion Agent
instagram_converter = Agent(
    role="Instagram Post Creator",
    goal=(
        "Convert the blog post to an engaging Instagram post based on {blog_content}."
        "Make it short, visually appealing, and use emojis and hashtags appropriately. "
        "Ensure it aligns with the style in sample_instagram.txt."
    ),
    verbose=True,
    memory=True,
    backstory="A creative AI specializing in summarizing blogs into visually appealing Instagram posts.",
    # New tools with file paths provided to FileReadTool
    tools=[FileReadTool(file_path='samples/sample_instagram.txt'), FileWriterTool()]  
)

# LinkedIn Conversion Agent
linkedin_converter = Agent(
    role="LinkedIn Post Creator",
    goal=(
        "Convert the blog post to a professional LinkedIn post based on {blog_content}."
        "Focus on detailed insights and a polished tone with minimal use of emojis. "
        "Ensure it aligns with the style in sample_linkedin.txt."
    ),
    verbose=True,
    memory=True,
    backstory="A professional AI that distills blog content into insightful LinkedIn posts.",
    # New tools with file paths provided to FileReadTool
    tools=[FileReadTool(file_path='samples/sample_linkedin.txt'), FileWriterTool()]  
)

# Define tasks
retrieval_task = Task(
    description="Retrieve information relevant to {blog_topic} from Chroma DB.",
    expected_output="Summarized insights related to {blog_topic}.",
    agent=data_retriever
)

writing_task = Task(
    description="Write a comprehensive blog post on {blog_topic}.",
    expected_output="A Markdown blog post on {blog_topic}.",
    agent=blog_writer,
    async_execution=False,
    output_file="outputs/blog_chroma_collection_name.txt"
)

instagram_task = Task(
    description="Convert the blog post to an Instagram-friendly format.",
    expected_output="Instagram post text based on the blog content.",
    agent=instagram_converter,
    async_execution=False,
    output_file="outputs/instagram_post_chroma_collection_name.txt"
)

linkedin_task = Task(
    description="Convert the blog post to a LinkedIn-friendly format.",
    expected_output="LinkedIn post text based on the blog content.",
    agent=linkedin_converter,
    async_execution=False,
    output_file="outputs/linkedin_post_chroma_collection_name.txt"
)

# Crew Definition
crew = Crew(
    agents=[data_retriever, blog_writer, instagram_converter, linkedin_converter],
    tasks=[retrieval_task, writing_task, instagram_task, linkedin_task],
    process=Process.sequential
)

# API: Generate Blog
@app.route('/generate_blog', methods=['POST'])
def generate_blog():
    try:
        data = request.get_json()
        blog_topic = data.get('blog_topic')
        folder_name = data.get('folder_name')
        file_name = data.get('file_name')
        if not blog_topic or not folder_name or not file_name:
            return jsonify({"error": "Missing 'blog_topic', 'folder_name', or 'file_name' parameter"}), 400

        # Prepare output file path
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)  # Ensure 'outputs' directory exists
        blog_output_file = os.path.join(output_dir, f"blog_{file_name}.txt")
        writing_task.output_file = blog_output_file

        # Define a specific crew for blog generation
        blog_crew = Crew(
            agents=[data_retriever, blog_writer],
            tasks=[retrieval_task, writing_task],
            process=Process.sequential
        )

        # Execute the blog generation process
        blog_crew.kickoff(inputs={'blog_topic': blog_topic, 'folder_name': folder_name, 'file_name': file_name})

        # Verify output file existence
        if not os.path.exists(blog_output_file):
            return jsonify({"error": "Blog output file not generated"}), 500

        # Read and return the content of the blog file
        blog_content = read_file_content(blog_output_file)
        return jsonify({
            "message": "Blog generated successfully",
            "blog_file": blog_output_file,
            "blog_content": blog_content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# API: Generate Instagram Post
@app.route('/generate_instagram', methods=['POST'])
def generate_instagram():
    try:
        data = request.get_json()
        blog_topic = data.get('blog_topic')
        folder_name = data.get('folder_name')
        file_name = data.get('file_name')

        if not blog_topic or not folder_name or not file_name:
            return jsonify({"error": "Missing 'blog_topic', 'folder_name', or 'file_name' parameter"}), 400

        # Determine file paths dynamically
        blog_output_file = os.path.join("outputs", f"blog_{file_name}.txt")
        instagram_output_file = os.path.join("outputs", f"instagram_post_{file_name}.txt")

        # Check if the blog file exists
        if not os.path.exists(blog_output_file):
            return jsonify({"error": "Blog file not found. Generate the blog first."}), 400

        # Read the blog content from the file
        blog_content = read_file_content(blog_output_file)
        if not blog_content:
            return jsonify({"error": "Blog content could not be read from the file."}), 500

        # Use the Instagram agent to generate the Instagram content
        instagram_task.output_file = instagram_output_file

        # Define the Crew and run the task (use 'kickoff' for execution)
        instagram_crew = Crew(
            agents=[instagram_converter],
            tasks=[instagram_task],
            process=Process.sequential
        )

        # Execute the crew with the blog content as input
        instagram_crew.kickoff(inputs={"blog_content": blog_content, 'folder_name': folder_name, 'file_name': file_name})

        # Check if the Instagram output file is generated
        if not os.path.exists(instagram_output_file):
            return jsonify({"error": "Instagram output file not generated"}), 500

        # Read the Instagram content from the file
        instagram_content = read_file_content(instagram_output_file)

        # Return the Instagram content and file path
        return jsonify({
            "message": "Instagram post generated successfully",
            "instagram_file": instagram_output_file,
            "instagram_content": instagram_content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_linkedin', methods=['POST'])
def generate_linkedin():
    try:
        data = request.get_json()
        blog_topic = data.get('blog_topic')
        folder_name = data.get('folder_name')
        file_name = data.get('file_name')

        if not blog_topic or not folder_name or not file_name:
            return jsonify({"error": "Missing 'blog_topic', 'folder_name', or 'file_name' parameter"}), 400

        # Determine file paths dynamically
        blog_output_file = os.path.join("outputs", f"blog_{file_name}.txt")
        linkedin_output_file = os.path.join("outputs", f"linkedin_post_{file_name}.txt")

        # Check if the blog file exists
        if not os.path.exists(blog_output_file):
            return jsonify({"error": "Blog file not found. Generate the blog first."}), 400

        # Read the blog content from the file
        blog_content = read_file_content(blog_output_file)
        if not blog_content:
            return jsonify({"error": "Blog content could not be read from the file."}), 500

        # Use the LinkedIn agent to generate the LinkedIn content
        linkedin_task.output_file = linkedin_output_file

        # Define the Crew and run the task (use 'kickoff' for execution)
        linkedin_crew = Crew(
            agents=[linkedin_converter],
            tasks=[linkedin_task],
            process=Process.sequential
        )

        # Execute the crew with the blog content as input
        linkedin_crew.kickoff(inputs={"blog_content": blog_content, 'folder_name': folder_name, 'file_name': file_name})

        # Check if the LinkedIn output file is generated
        if not os.path.exists(linkedin_output_file):
            return jsonify({"error": "LinkedIn output file not generated"}), 500

        # Read the LinkedIn content from the file
        linkedin_content = read_file_content(linkedin_output_file)

        # Return the LinkedIn content and file path
        return jsonify({
            "message": "LinkedIn post generated successfully",
            "linkedin_file": linkedin_output_file,
            "linkedin_content": linkedin_content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/get_files', methods=['GET'])
def get_files():
    """
    Optimized API endpoint to fetch file collections.
    """
    base_dir = request.args.get('data_type')

    if not os.path.exists(base_dir):
        return jsonify({"error": f"The folder '{base_dir}' does not exist."}), 404

    # Use scandir to iterate through directories faster
    files = [f"{entry.name}.pdf" for entry in scandir(base_dir) if entry.is_dir()]

    return jsonify(files)

# API endpoint to load ChromaDB
@app.route('/load_chroma', methods=['POST'])
def load_chroma():
    try:
        # Get the folder_name and file_name from the request body
        data = request.json
        folder_name = data.get('folder_name')
        file_name = data.get('file_name')
        
        if not folder_name or not file_name:
            return jsonify({"error": "folder_name and file_name are required"}), 400
        
        # Initialize ChromaDB
        chroma_db = init_chroma_db(folder_name, file_name)
        
        return jsonify({"message": f"ChromaDB initialized at {folder_name}/{file_name}"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def post_to_linkedin_api(linkedin_post_content=None, linkedin_image_urls=None):
    """
    Posts content (with multiple image URLs) to LinkedIn using LinkedIn's API.
    """
    api_url = 'https://api.linkedin.com/v2/ugcPosts'
    access_token = linkedin_access_token

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'X-Restli-Protocol-Version': '2.0.0'
    }

    # Step 1: Fetch user information to get the LinkedIn URN (unique identifier for the user)
    response = requests.get('https://api.linkedin.com/v2/userinfo', headers=headers)
    if response.status_code == 200:
        user_info = response.json()
        urn = user_info.get('sub')  # User's LinkedIn URN
    else:
        return {"error": f"Failed to fetch user info: {response.status_code}, {response.text}"}

    author = f'urn:li:person:{urn}'

    # Step 2: Build the base structure for the LinkedIn post
    post_data = {
        "author": author,
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {
                    "text": linkedin_post_content if linkedin_post_content else ''
                },
                "shareMediaCategory": "NONE"  # Default to no media
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        }
    }

    # Step 3: Handle image uploads from the URLs (if any)
    asset_urns = []

    if linkedin_image_urls:
        for image_url in linkedin_image_urls:
            try:
                # Fetch the image binary from the URL
                image_response = requests.get(image_url)
                image_response.raise_for_status()  # Ensure the URL is valid
                image_binary = image_response.content

                # Register the upload URL with LinkedIn
                media_upload_url = 'https://api.linkedin.com/v2/assets?action=registerUpload'
                image_upload_request_data = {
                    "registerUploadRequest": {
                        "owner": author,
                        "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                        "serviceRelationships": [
                            {
                                "identifier": "urn:li:userGeneratedContent",
                                "relationshipType": "OWNER"
                            }
                        ],
                        "supportedUploadMechanism": ["SYNCHRONOUS_UPLOAD"]
                    }
                }

                upload_response = requests.post(media_upload_url, headers=headers, json=image_upload_request_data)
                if upload_response.status_code == 200:
                    upload_info = upload_response.json()
                    upload_url = upload_info['value']['uploadMechanism']['com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest']['uploadUrl']
                    asset_urn = upload_info['value']['asset']

                    # Upload the binary image to LinkedIn
                    image_headers = {
                        'Authorization': f'Bearer {access_token}',
                        'Content-Type': 'application/octet-stream'
                    }
                    image_upload_response = requests.post(upload_url, headers=image_headers, data=image_binary)
                    if image_upload_response.status_code == 201:
                        asset_urns.append({
                            "status": "READY",
                            "media": asset_urn,
                            "title": {
                                "text": "Uploaded Image"
                            }
                        })
                    else:
                        return {"error": f"Failed to upload image: {image_upload_response.status_code}, {image_upload_response.text}"}
                else:
                    return {"error": f"Failed to register image upload: {upload_response.status_code}, {upload_response.text}"}
            except requests.RequestException as e:
                return {"error": f"Failed to fetch image from URL: {str(e)}"}

    # Step 4: Add images to the post (if any were successfully uploaded)
    if asset_urns:
        post_data["specificContent"]["com.linkedin.ugc.ShareContent"]["shareMediaCategory"] = "IMAGE"
        post_data["specificContent"]["com.linkedin.ugc.ShareContent"]["media"] = asset_urns

    # Step 5: Post the content to LinkedIn
    post_response = requests.post(api_url, headers=headers, data=json.dumps(post_data))
    if post_response.status_code == 201:
        return {"message": "Post published to LinkedIn successfully!"}
    else:
        return {"error": f"Failed to post to LinkedIn: {post_response.status_code}, {post_response.text}"}


@app.route('/post_to_linkedin', methods=['POST'])
def post_to_linkedin():
    """
    API endpoint to handle posting content (with optional multiple images) to LinkedIn.
    Expects a JSON payload:
    {
        "content": "Post content",
        "images": ["ImageURL1", "ImageURL2"]
    }
    """
    try:
        data = request.get_json()
        content = data.get('content', None)
        image_urls = data.get('images', None)

        # Validate image_urls is a list
        if image_urls and not isinstance(image_urls, list):
            return jsonify({"error": "The 'images' field must be a list of URLs."}), 400

        # Call the LinkedIn API helper function
        result = post_to_linkedin_api(linkedin_post_content=content, linkedin_image_urls=image_urls)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Azure configuration
connection_string = "DefaultEndpointsProtocol=https;AccountName=genai123;AccountKey=EstDEzRJ+CP3SMlRlurcbK1cn8JAxoHT2VKfLwsZb+SZavm3p6uB8LtY2UqsTmskJYVK4BpdxmNm+AStjNtfIQ==;EndpointSuffix=core.windows.net"
container_name = "proposal"

# Directories for Azure data and Chroma DB
azure_data_folder = "Azure_data"
azure_chroma_folder = "azure_chroma_db"


# Utility: Clear a folder (updated)
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        if not os.listdir(folder_path):  # Check if folder is empty
            print(f"Folder is already empty: {folder_path}")
        else:
            shutil.rmtree(folder_path)  # Clear existing contents
            print(f"Cleared folder: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)


# Function: Download PDFs from Azure Blob Storage
def download_pdfs_from_azure(connection_string, container_name, download_folder):
    print("Downloading PDFs from Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    for blob in container_client.list_blobs():
        if blob.name.endswith(".pdf"):
            blob_client = container_client.get_blob_client(blob.name)
            download_path = os.path.join(download_folder, os.path.basename(blob.name))
            with open(download_path, "wb") as file:
                file.write(blob_client.download_blob().readall())
            print(f"Downloaded: {blob.name}")

# Function to extract Table of Contents (TOC)
def extract_toc_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    toc_content = ""

    for page_num in range(5):  # First 5 pages
        try:
            page = doc[page_num]
            toc_content += page.get_text()
        except IndexError:
            break

    prompt_template = """
    The following text may contain a Table of Contents (TOC). Please analyze the content and determine if a TOC is present. whole toc content as below sample format. Format the start and end page number as integer. 
    Capital letters are the main headings, and lower case letters are subheadings. Titles with subheadings shouldn't be set to the same page number.
    
    TOC is found, return it in the following format without any additional text and without any \n values ::
    [
        ("SECTION NAME", start_page_number, end_page_number),
        ...
    ]
    If no TOC is present, return "No".

    Text to analyze:
    {toc_content}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["toc_content"], template=prompt_template)
    chain = prompt | llm

    try:
        response = chain.invoke({"toc_content": toc_content})
        toc_list = ast.literal_eval(response)
    except Exception as e:
        print(f"Error parsing TOC content: {e}")
        toc_list = "No"
    finally:
        doc.close()

    return toc_list


# Function to process PDF and create a DataFrame
def panda_dataframe(pdf_path, toc_list):
    df = pd.DataFrame(columns=["Topic", "Content", "Start Page", "End Page"])
    doc = fitz.open(pdf_path)

    for topic, start_page, end_page in toc_list:
        content = ""
        for page_num in range(start_page - 1, end_page):  # Page numbering in PyMuPDF starts at 0
            try:
                page = doc[page_num]
                content += page.get_text()
            except IndexError:
                break

        new_row = pd.DataFrame({
            "Topic": [topic],
            "Content": [content],
            "Start Page": [start_page],
            "End Page": [end_page]
        })
        df = pd.concat([df, new_row], ignore_index=True)

    doc.close()
    return df


# Function to split text into chunks
def process_row(row):
    topic = row["Topic"]
    content = row["Content"]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(content)
    return [f"{topic}: {chunk}" for chunk in chunks]


# Function to save text chunks to Chroma DB
def dataframe_to_chunking(pdf_path, dataframe, chroma_folder):
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    persist_directory = os.path.join(chroma_folder, file_name)  # Create a folder per file
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    for idx, row in dataframe.iterrows():
        chunks_with_topic = process_row(row)
        for chunk in chunks_with_topic:
            db.add_texts([chunk], metadatas=[{
                "topic": row["Topic"],
                "start_page": row["Start Page"],
                "end_page": row["End Page"]
            }])
    print(f"Chroma DB stored at: {persist_directory}")


# Function to process text without TOC
def text_to_chunks(pdf_path, chroma_folder):
    doc = fitz.open(pdf_path)
    content = ""
    for page_num in range(len(doc)):
        content += doc[page_num].get_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(content)

    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    persist_directory = os.path.join(chroma_folder, file_name)  # Create a folder per file
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    for chunk in chunks:
        db.add_texts([chunk], metadatas={"file_name": file_name})
    print(f"Chroma DB stored at: {persist_directory}")


# Main processing function
def process_folder(folder_path, chroma_folder):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Processing {pdf_path}...")

        toc_list = extract_toc_from_pdf(pdf_path)
        if toc_list != "No":
            dataframe = panda_dataframe(pdf_path, toc_list)
            dataframe_to_chunking(pdf_path, dataframe, chroma_folder)
        else:
            text_to_chunks(pdf_path, chroma_folder)

@app.route('/process', methods=['POST'])
def process_data():
    # Check and handle empty or non-empty folders
    if not os.listdir(azure_data_folder):
        print("Azure data folder is empty, proceeding with download and processing.")
    else:
        print("Azure data folder has data, clearing it.")
        clear_folder(azure_data_folder)

    if not os.listdir(azure_chroma_folder):
        print("Azure Chroma DB folder is empty, proceeding with storage.")
    else:
        print("Azure Chroma DB folder has data, clearing it.")
        clear_folder(azure_chroma_folder)

    # Step 2: Download PDFs from Azure Blob Storage
    download_pdfs_from_azure(connection_string, container_name, azure_data_folder)

    # Step 3: Process PDFs and store in Chroma DB
    process_folder(azure_data_folder, azure_chroma_folder)

    return jsonify({"message": "Processing completed successfully!"})

if __name__ == '__main__':
    app.run(debug=True)
