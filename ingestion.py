
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

# Set the path to the data directory
DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    """
    Creates a FAISS vector store from documents in the data directory.
    """
    # Create a DirectoryLoader for each tenant
    loaders = []
    for tenant in os.listdir(DATA_PATH):
        tenant_path = os.path.join(DATA_PATH, tenant)
        if os.path.isdir(tenant_path):
            # Use a generic loader and add metadata
            loader = DirectoryLoader(tenant_path, glob="*.txt", show_progress=True)
            docs = loader.load()
            for doc in docs:
                doc.metadata["company"] = tenant
            loaders.extend(docs)

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(loaders)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # Create and save the FAISS vector store
    db = FAISS.from_documents(splits, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store created at {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
