import glob
import os

from dotenv import load_dotenv
# from langchain_openai import OpenAIChat
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredImageLoader, DirectoryLoader, TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import gradio as gr



model = OllamaLLM(model="llama2", base_url="http://localhost:11434")
DB_NAME = "vector_database/chromadb/vector_db"
load_dotenv(override=True)

knowledge_path = "data/extracted_data/**/md/*_hydrated.md"
files = glob.glob(knowledge_path, recursive=True)
print(f"[INFO] Found {len(files)} markdown files for ingestion. Those files are {files}")

entire_knowledge_base = ""

# for file in files:
#     with open(file, "r", encoding="utf-8") as f:
#         entire_knowledge_base += f.read() + "\n\n"

# print(f"[INFO] Total characters in knowledge base: {len(entire_knowledge_base)}")

# encoding = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
# tokens = encoding.encode(entire_knowledge_base)
# print(f"[INFO] Total tokens in knowledge base: {len(tokens)}")


folders = glob.glob("data/extracted_data/**/md/")

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="*_hydrated.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    folder_docs = loader.load()
    for doc in folder_docs:
        print(f"[INFO] Loaded document: {doc.metadata['source']} with {len(doc.page_content)} characters.")
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"[INFO] Total documents loaded: {len(documents)}")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,
    length_function=len
)

chunks = text_splitter.split_documents(documents)

print(f"[INFO] Total chunks created: {len(chunks)}")


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(DB_NAME):
    print(f"[INFO] Loading existing vector store from {DB_NAME}...")
    Chroma(persist_directory=DB_NAME, embedding_function=embedding_model).delete_collection() 


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=DB_NAME
)

print(f"[INFO] Vector store created and persisted at {DB_NAME}.")
print(f"[INFO] Total vectors in store: {vector_store._collection.count()}")

collection = vector_store._collection
count = collection.count()
print(f"[INFO] Total vectors in the collection: {count}")

sample_embedding = collection.get(limit=1, include=["embeddings"])[ "embeddings" ][0]

dimension = len(sample_embedding)
print(f"[INFO] Dimension of the embeddings: {dimension}")


# vector_store = Chroma(
#     persist_directory=DB_NAME,
#     embedding_function=embedding_model
# )

# retriever = vector_store.as_retriever()
# llm = ChatOllama(model=model, temperature=0.2)

# # retriever.invoke("")