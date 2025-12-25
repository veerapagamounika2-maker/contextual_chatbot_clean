import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
# STEP 1: Load documents
# -------------------------------
DATA_FOLDER = "data"
documents = []

for file_name in sorted(os.listdir(DATA_FOLDER)):
    if file_name.endswith(".txt"):
        file_path = os.path.join(DATA_FOLDER, file_name)
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)

print(f"Documents loaded: {len(documents)}")

# -------------------------------
# STEP 2: Split into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print(f"Chunks created: {len(chunks)}")

# -------------------------------
# STEP 3: Create embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# STEP 4: Store vectors in FAISS
# -------------------------------
vector_db = FAISS.from_documents(chunks, embeddings)

# -------------------------------
# STEP 5: Save FAISS index
# -------------------------------
FAISS_INDEX_PATH = "faiss_index"
vector_db.save_local(FAISS_INDEX_PATH)

print("FAISS index created and saved successfully")
