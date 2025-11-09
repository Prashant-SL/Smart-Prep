# Smart Prep
> Your personal AI interview coach.

Smart Prep is an AI-powered application that helps you prepare for job interviews by analyzing your resume against a specific job description. It generates a list of probable interview questions and provides actionable suggestions to improve your resume, ensuring you're fully prepared to land the job.

---

## 🎯 Core Features

* **Dynamic Question Generation:** Generates interview questions tailored specifically to the job's requirements and your personal experience.
* **Resume Gap Analysis:** Intelligently identifies key skills and requirements from the job description that are *missing* from your resume.
* **AI-Powered Suggestions:** Provides actionable advice on how to update your resume to better match the role and pass automated screening.
* **PDF & Text Support:** Accepts resumes as both raw text (`/analyse-resume`) and direct PDF uploads (`/upload-resume`).
* **Modern RAG Pipeline:** Built using a cutting-edge Retrieval-Augmented Generation (RAG) pipeline for high-quality, context-aware results.

---

## ⚙️ How It Works: The RAG Pipeline

This project is more than just a simple prompt. It uses a complete RAG pipeline to generate highly relevant results.

1.  **Ingest & Chunk:** The resume and job description (JD) are cleaned and broken down into small, meaningful chunks (e.g., paragraphs, sections).
2.  **Embed:** The *resume* chunks are converted into numerical vectors (embeddings) using a `sentence-transformers` model.
3.  **Index:** These resume vectors are loaded into a high-speed, in-memory `faiss-cpu` vector store, making them searchable.
4.  **Retrieve (The "R" in RAG):** The app iterates through each *job description* chunk and treats it as a search query against the resume's vector store to find the most relevant matching piece of experience.
5.  **Generate (The "G" in RAG):** The `google/gemma-2-2b-it` model is used to generate text based on two distinct scenarios:
    * **For Matches:** An LLM prompt is sent: "The JD requires 'React Hooks,' and the resume mentions 'React projects.' Please generate a specific technical question about this."
    * **For Gaps:** When a JD requirement has no good match (a high distance score), a different prompt is sent: "The JD requires 'Docker,' but the resume doesn't mention it. Please generate a probing question *and* a resume improvement suggestion."
6.  **Respond:** The final lists of questions and suggestions are de-duplicated and returned to the user in a clean JSON format.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI
* **LLM:** `google/gemma-2-2b-it` (loaded via `transformers`)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Store:** `faiss-cpu`
* **Core Libraries:** `torch`, `bitsandbytes`, `pydantic`, `python-multipart`
* **PDF Handling:** `pypdf` (or similar utility)

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* `pip` and `venv`
* A CUDA-enabled GPU is highly recommended (for running the models)

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/smart-prep.git](https://github.com/your-username/smart-prep.git)
cd smart-prep
```

### 2. Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

### 3. Install Dependencies
Create a requirements.txt file with the core dependencies and install them.

requirements.txt

fastapi
uvicorn[standard]
torch
transformers
bitsandbytes
sentence-transformers
faiss-cpu
python-multipart
# Add your PDF extraction library, e.g., pypdf
pypdf
accelerate
pydantic

Install

pip install -r requirements.txt

### 4. Run the server
uvicorn main:app --host 0.0.0.0 --port 8000

