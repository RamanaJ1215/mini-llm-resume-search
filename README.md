# 🤖 Mini LLM Resume Search

A terminal-based AI tool that allows you to ask questions about your resume using OpenAI embeddings + GPT-4. It’s a simple RAG (Retrieval-Augmented Generation) project that loads your resume, chunks it, embeds it, finds the best match, and answers your questions smartly.

---

## 📌 Why I Built This

Inspired by a [YouTube podcast](https://www.youtube.com/watch?v=RDS4Crfk_wQ&t=2442s) featuring **Manish Surapaneni** and **Jignesh Talasila** on the **HeadRock Show**, I heard:

> 💡 "If someone can build embedding, vector db, and a small LLM project using AI APIs — we’ll provide opportunities.”

So I took the first step. As a .NET Full Stack Developer, I decided to explore the AI space and showcase it through this mini project.

---

## 🚀 Setup & Run Instructions

### ✅ Step-by-step

```bash
# 1. Clone this repo
git clone https://github.com/RamanaJ1215/mini-llm-resume-search.git
cd mini-llm-resume-search

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add your OpenAI API key
# Create a .env file with this content:
echo OPENAI_API_KEY=your_openai_key_here > .env

# 6. Add your resume
# Create or paste your resume content into a file named:
# resume.txt

# 7. Run the app
python appl.py

#Project folder structure
mini-llm-resume-search/
├── appl.py                 # Main logic (embedding, similarity, Q&A)
├── resume.txt              # Your resume in plain text format
├── resume_data.pkl         # Pickled cache of embeddings
├── .env                    # Contains your OpenAI API Key
├── requirements.txt        # All required packages
├── README.md               # This file
└── screenshots/            # Screenshot assets
    ├── code1.png
    ├── code2.png
    └── output.png


## 🔮 Next Steps

- ✅ Add FAISS for vector database and fast search
- ✅ Create web UI using Streamlit
- 🔄 Support multi-resume comparison
- 📄 Resume vs JD matching (score + improvement tips)
- 🌐 Deploy to the web and make public

## 🙏 Acknowledgments

This project was inspired by a podcast from the [HeadRock Show](https://www.youtube.com/@HeadRockShow) where **Manish Surapaneni** encouraged devs to explore embeddings, vector search, and LLMs to prepare for the future of AI.

---

## ✨ Connect With Me

- 💼 [LinkedIn](www.linkedin.com/in/ramana-j-504363142)
- 👨‍💻 [GitHub](https://github.com/RamanaJ1215/mini-llm-resume-search.git)
