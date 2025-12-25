import os
from openai import OpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------
# STEP 1: Load embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# STEP 2: Load FAISS index
# -------------------------------
FAISS_INDEX_PATH = "faiss_index"

vector_db = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded")

# -------------------------------
# STEP 3: OpenAI client
# (DIRECT KEY – GUARANTEED)
# -------------------------------
client = OpenAI(
    api_key="sk-PASTE-YOUR-REAL-API-KEY-HERE"
)

# -------------------------------
# STEP 4: Ask questions (RAG)
# -------------------------------
def ask_question(query):
    docs = vector_db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful customer support assistant.

Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# -------------------------------
# STEP 5: Interactive chatbot
# -------------------------------
print("\nCustomer Support Chatbot (type 'exit' to quit)\n")

while True:
    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit"]:
        print("Bot: Thank you! Have a nice day.")
        break

    answer = ask_question(user_query)
    print(f"\nBot: {answer}\n")
