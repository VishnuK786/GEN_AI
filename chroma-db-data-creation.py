import os
import fitz  # PyMuPDF
print(fitz.__doc__)
import pandas as pd
from dotenv import load_dotenv
import ast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings,AzureChatOpenAI
from langchain_chroma import Chroma
from langtrace_python_sdk import langtrace

# Load environment variables
load_dotenv()
langtrace.init(api_key = os.getenv("LANGTRACE_API_KEY"))
api_type = 2  # 1 - Azure OpenAI, 2 - OpenAI

if api_type == 1:
    azure_endpoint = os.getenv("Azure_API_ENDPOINT")
    api_key = os.getenv("AZURE_API_KEY")
    api_version = "2023-03-15-preview"
    deployment_name = "gpt-4o"  # Ensure this matches your deployment name in Azure

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
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(max_tokens=300)


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


# Directories for local and Azure data
local_data_folder = "local_data"
azure_data_folder = "Azure_data"

# Output folders for Chroma DB
local_chroma_folder = "local_chroma_db"
azure_chroma_folder = "azure_chroma_db"

# Process both folders
print("Processing local data...")
process_folder(local_data_folder, local_chroma_folder)

print("Processing Azure data...")
process_folder(azure_data_folder, azure_chroma_folder)
