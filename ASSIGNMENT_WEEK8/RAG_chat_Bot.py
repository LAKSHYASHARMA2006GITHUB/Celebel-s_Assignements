import pandas as pd
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load data
df = pd.read_csv("Training Dataset.csv").fillna("unknown")
documents = [f"Applicant {i}: {row.to_dict()}" for i, row in df.iterrows()]

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents)

# Build FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# Load generator model (Hugging Face light model)
generator = pipeline(
    "text2text-generation",  # <- This is different from "text-generation"
    model="google/flan-t5-base"
)

# Retrieval function
def retrieve(query, top_k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, top_k)
    return [documents[i] for i in indices[0]]

# RAG generation
def generate_answer(query, retrieved_docs):
    context = "\n".join(retrieved_docs)
    prompt = f"""Context:
{context}

Question: {query}
Answer:"""
    response = generator(prompt)[0]["generated_text"]
    return response.split("Answer:")[-1].strip()

# Streamlit UI
st.title("📊 Loan Approval RAG Chatbot")
question = st.text_input("Ask a question about the loan dataset")

if question:
    with st.spinner("Thinking..."):
        context_docs = retrieve(question)
        answer = generate_answer(question, context_docs)
        st.markdown(f"**Answer:** {answer}")
        with st.expander("🔍 Retrieved Context"):
            for doc in context_docs:
                st.write(doc)
