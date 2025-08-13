# 🎯 AI Interview Practice App

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?logo=streamlit)](https://placement-prep-model-bk4kxhhtmld4xml4wkrx8j.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🚀 **An AI-powered interactive interview preparation platform** built with **Streamlit**, **LangChain**, and **LLMs** to help you master technical & HR interview rounds.

---

## 📌 Overview

This app simulates a **real interview experience**:
- Choose your **domain** (DSA, AI, System Design, HR, Software Engineer).
- Get **dynamic AI-curated questions**.
- Answer in chat format.
- Receive **instant feedback** and **grading** based on ideal answers.
- Practice **code-specific DSA questions** in your preferred language *(C++, Java, Python)*.

Live Demo: 👉 **[Click here to try it](https://placement-prep-model-bk4kxhhtmld4xml4wkrx8j.streamlit.app/)**

---

## ✨ Features

### 🎯 Domain-Based Practice
- **DSA** – Solve coding problems with language filtering.
- **AI** – Machine learning & AI concepts.
- **Software Engineer Round** – General tech interview questions.
- **System Design** – Large-scale architecture problems.
- **HR Round** – Behavioral & soft skills.

### 🤖 AI-Powered Q&A
- Uses **LLaMA 3** through LangChain for realistic responses.
- **Conversational memory** retains context.
- **Streaming output** for real-time typing effect.

### 🧠 Intelligent Grading
- Compares your answer with the **ideal solution**.
- Grades correctness.
- Provides **feedback** + **encouragement**.

### 🛠 DSA Language Filtering
- Choose your preferred coding language.
- Automatically filters questions with solutions in **C++**, **Java**, or **Python**.

---

## 🖥️ Screenshots

| Home Screen | Interview Question | AI Feedback |
|-------------|-------------------|-------------|
| ![Home](assets/home.png) | ![Question](assets/question.png) | ![Feedback](assets/feedback.png) |

*(Replace with your own screenshots)*

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend AI**: [LangChain](https://www.langchain.com/)
- **LLM**: LLaMA 3 (via Groq API)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **Datasets**: HuggingFace datasets

---

## ⚙️ Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/hrk022/ai-interview-practice.git
cd ai-interview-practice

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Add your API key to open_ai.env
echo "OPENAI_API_KEY=your_api_key_here" > open_ai.env

# 4️⃣ Run the app
streamlit run streamlit_interview_app.py
