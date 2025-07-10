# PDF Question Answering using RAG + Groq LLaMA3 + FastAPI

This project allows you to ask questions over a PDF using Retrieval-Augmented Generation (RAG) with Groq's LLaMA3 model. It returns a structured answer along with confidence, source page, and chunk size.

---

# Features

- Load and split PDF into chunks
- Embed chunks using Sentence Transformers
- Store embeddings in FAISS vector store
- Retrieve top relevant chunks using cosine similarity
- Use Groq LLaMA3-8B model to generate concise answers
- Returns: answer, source page, confidence score, and chunk size

---
install below:

pip install -r requirements.txt

##  Project Structure

```bash
├── app.py                        # FastAPI backend
├── 10050-medicare-and-you_0.pdf # PDF file to query
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```


groq_api_key="your_actual_groq_api_key"
 #Here you can replace with your api key


---
 
## Run the App

Run this command in the terminal:

```bash
uvicorn app:app --reload
```


## Technologies Used:

-Groq LLaMA3-8B via langchain_groq
-Sentence Transformers (all-MiniLM-L6-v2
-FAISS for vector search
-LangChain
-FastAPI






