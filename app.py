# Importing Libraries

import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
#from dotenv import load_dotenv


#load_dotenv()
# --- Here i am using direct API Key instead or dotenv ---

# Initialize FastAPI
app = FastAPI()

# ----- Load PDF -----

loader = PyPDFLoader(r"10050-medicare-and-you_0.pdf")
pages = loader.load_and_split()

# ----- Chunking - Splitting PDF into chunks-----

lengths = [len(p.page_content) for p in pages]
chunk_size = int(np.mean(lengths) + np.std(lengths) / 2)
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
chunks = splitter.split_documents(pages)

# ----- Embedding and creating vector DB -----

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ----- Groq LLM ----- using llama3-8b-8192 LLM Model

llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.5,
    max_tokens=512,
    groq_api_key="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"
)

# ----- Custom Prompt for Concise Answer -----

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Based only on the context below, answer the question clearly in sentences. 
Be direct, avoid bullet points or markdown formatting.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ----- RAG Chain with Custom Prompt -----

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

# ----- Input Model -----

class QueryInput(BaseModel):
    query: str

# ----- API Endpoint -----

@app.post("/ask")
async def ask(data: QueryInput):
    query = data.query.strip()
    if not query:
        return {
            "answer": "Query cannot be empty.",
            "source_page": -1,
            "confidence_score": 0.0,
            "chunk_size": 0
        }

    result = rag_chain(query)
    docs = result.get("source_documents", [])

    if not docs:
        return {
            "answer": "No relevant content found.",
            "source_page": -1,
            "confidence_score": 0.0,
            "chunk_size": 0
        }

    doc_text = docs[0].page_content or ""
    query_vec = embedding_model.embed_query(query)
    doc_vec = embedding_model.embed_query(doc_text)
    score = cosine_similarity([query_vec], [doc_vec])[0][0]

    return {
        "answer": result.get("result", "No answer generated."),
        "source_page": docs[0].metadata.get("page",-1),
        "confidence_score": round(float(score), 2),
        "chunk_size": len(doc_text)
    }
