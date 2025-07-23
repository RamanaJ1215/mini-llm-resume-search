import os
import sys
import io
import pickle
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv
from openai import OpenAI

# Ensure UTF-8 output for proper terminal rendering
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY not found in .env file")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Utility: Chunk text into paragraphs
def chunk_text(text):
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks if chunks else [text.strip()]

# Utility: Get OpenAI embedding
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=text, model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        raise RuntimeError(f"Error getting embedding: {e}")

# Utility: Cosine similarity
def cosine_similarity(query_vec, matrix):
    query_norm = norm(query_vec)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    dot_products = matrix @ query_vec
    return dot_products / (query_norm * matrix_norms + 1e-10)

# Resume QA Engine
def ask_resume(question, resume_chunks, resume_embeddings):
    query_emb = get_embedding(question)
    similarities = cosine_similarity(query_emb, np.array(resume_embeddings))
    top_idx = np.argmax(similarities)
    top_chunk = resume_chunks[top_idx]

    prompt = f"""
You are a professional resume assistant. Based on the following resume section, answer the question concisely and clearly.

Resume Section:
\"\"\"{top_chunk}\"\"\"

Question:
\"\"\"{question}\"\"\"

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about resumes."},
                {"role": "user", "content": prompt.strip()}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Failed to generate answer: {e}"

# Save resume chunks and embeddings
def save_embeddings(chunks, embeddings, filename="resume_data.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((chunks, embeddings), f)

# Load resume chunks and embeddings
def load_embeddings(filename="resume_data.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Main runner
if __name__ == "__main__":
    try:
        with open("resume.txt", "r", encoding="utf-8") as f:
            resume_text = f.read()
    except FileNotFoundError:
        print("❌ 'resume.txt' not found.")
        sys.exit(1)

    resume_pickle = "resume_data.pkl"

    if os.path.exists(resume_pickle):
        resume_chunks, resume_embeddings = load_embeddings(resume_pickle)
        print(f"✅ Loaded {len(resume_chunks)} chunks from cache.")
    else:
        print("📄 Chunking resume...")
        resume_chunks = chunk_text(resume_text)

        print("🔄 Getting embeddings for each section...")
        try:
            resume_embeddings = [get_embedding(chunk) for chunk in resume_chunks]
        except Exception as e:
            print(f"❌ Failed to compute embeddings: {e}")
            sys.exit(1)

        save_embeddings(resume_chunks, resume_embeddings, resume_pickle)
        print(f"✅ Embedded and saved {len(resume_chunks)} chunks successfully.")

    # Start Q&A loop
    while True:
        question = input("\nAsk me a question about the resume (type 'exit' or 'quit' to stop): ").strip()
        if question.lower() in {'exit', 'quit'}:
            print("👋 Goodbye!")
            break
        if not question:
            print("❗ Please enter a valid question.")
            continue
        answer = ask_resume(question, resume_chunks, resume_embeddings)
        print("\nAnswer:", answer)
