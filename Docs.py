from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter , Language
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.document_loaders import PythonLoader

def get_youtube_docs(document):
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,)
    docs = text_splitter.split_documents(documents=[document])
    return docs

def docs_textFile(filePath):
    text_loader = TextLoader(filePath)
    documets = text_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,)
    docs = text_splitter.split_documents(documents=documets)
    return docs


def docs_PDFFile(filePath):
    pdf_loader = PyMuPDFLoader(file_path=filePath)
    documets = pdf_loader.load()
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,)
    docs = pdf_splitter.split_documents(documents=documets)
    return docs

def docs_HTMLFile(filePath):
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    html_header_splits = html_splitter.split_text_from_file(file=filePath)
    html_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,)
    docs = html_splitter.split_documents(html_header_splits)
    return docs

def docs_MDFile(filePath):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    try:
        with open(filePath, 'r') as file:
            file_content = file.read()
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_header_splits = markdown_splitter.split_text(file_content)
            md_splitter = RecursiveCharacterTextSplitter(
                chunk_size=150,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,)
            docs = md_splitter.split_documents(md_header_splits)
        return docs
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)


def docs_pythonFile(filePath):
    loader = PythonLoader(file_path=filePath)
    documets = loader.load()
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, 
        chunk_size=400,
        chunk_overlap=50 , 
        length_function=len,
    )
    docs = python_splitter.split_documents(documents=documets)
    return docs


def get_docs(filepath,fileName):
    get_file_type = fileName.split(".")[-1]
    if get_file_type == "py":
        docs = docs_pythonFile(filePath=filepath)
        return docs
    elif get_file_type == "pdf":
        docs = docs_PDFFile(filePath=filepath)
        return docs
    elif get_file_type == "txt":
        docs = docs_textFile(filePath=filepath)
        return docs
    elif get_file_type == "md":
        docs = docs_MDFile(filePath=filepath)
        return docs
    elif get_file_type == "html":
        docs = docs_HTMLFile(filePath=filepath)
        return docs
    return []

