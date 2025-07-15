import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Title
st.title("📚 RAG Q&A Chatbot")
st.write("Ask any question based on the uploaded dataset (`Training Dataset.csv`)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Training Dataset.csv")
    docs = df.astype(str).apply(lambda x: ' | '.join(x), axis=1).tolist()
    return docs

docs = load_data()

# Load embedding model
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = get_embedder()

# Create or load FAISS index
@st.cache_resource
def build_faiss_index(docs):
    embeddings = embedder.encode(docs, convert_to_tensor=False)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

index, embeddings = build_faiss_index(docs)

# Load generation model
@st.cache_resource
def get_generator():
    return pipeline("text-generation", model="tiiuae/falcon-rw-1b", tokenizer="tiiuae/falcon-rw-1b", max_new_tokens=200)

generator = get_generator()

# Query function
def rag_answer(question, top_k=3):
    q_embedding = embedder.encode([question])
    D, I = index.search(np.array(q_embedding), top_k)
    context = "\n".join([docs[i] for i in I[0]])
    prompt = f"Answer the following question using the context.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = generator(prompt)[0]['generated_text']
    return response.split("Answer:")[-1].strip()

# User input
user_question = st.text_input("❓ Enter your question:")

if user_question:
    with st.spinner("Generating answer..."):
        response = rag_answer(user_question)
        st.markdown("### 💬 Answer")
        st.success(response)
