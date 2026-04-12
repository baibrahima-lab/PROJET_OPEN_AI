import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import CONFIG
from typing import List

def ingest_documents(data_path: str = None, persist_directory: str = None):
    """
    Pipeline d'ingestion intelligent avec métadonnées enrichies.
    Supporte PDF, TXT, MD.
    """
    data_path = data_path or CONFIG.DATA_PATH
    persist_directory = persist_directory or CONFIG.VECTORSTORE_PATH
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"📁 Dossier {data_path} créé. Ajoutez des documents et relancez.")
        return
    
    print(f"--- 📂 Chargement des documents depuis {data_path} ---")
    
    # Chargement multi-format
    documents = []
    
    # PDF
    if any(f.endswith('.pdf') for f in os.listdir(data_path)):
        pdf_loader = DirectoryLoader(
            data_path, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents.extend(pdf_loader.load())
    
    # Texte/Markdown
    for ext in ['*.txt', '*.md']:
        text_loader = DirectoryLoader(
            data_path,
            glob=f"**/{ext}",
            loader_cls=TextLoader,
            show_progress=True
        )
        try:
            documents.extend(text_loader.load())
        except:
            pass
    
    if not documents:
        print("⚠️ Aucun document trouvé")
        return
    
    print(f"✅ {len(documents)} documents chargés")
    
    # Chunking intelligent
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.CHUNK_SIZE,
        chunk_overlap=CONFIG.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️  {len(chunks)} chunks créés")
    
    # Enrichissement des métadonnées
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'unknown')
        chunk.metadata['doc_type'] = os.path.splitext(source)[1]
        chunk.metadata['chunk_id'] = i
    
    # Création des embeddings
    embeddings = OpenAIEmbeddings(
        model=CONFIG.EMBEDDING_MODEL,
        api_key=CONFIG.OPENAI_API_KEY
    )
    
    # Stockage avec batching
    print("--- 🧠 Indexation vectorielle (ChromaDB) ---")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    vectorstore.persist()
    print(f"--- ✅ Indexation terminée: {persist_directory} ---")
    print(f"   Collection: {vectorstore._collection.name}")
    print(f"   Documents: {vectorstore._collection.count()}")

if __name__ == "__main__":
    ingest_documents()