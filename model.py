# streamlit_interview_app.py
import os
import streamlit as st
import json
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler
import random
from typing import List, Tuple
import re

# ---------- Configuration ----------
load_dotenv("open_ai.env")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key 
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# ---------- Streaming callback handler ----------
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        try:
            self.container.markdown(self.text + "▌")
        except Exception:
            self.container.write(self.text + "▌")

# ---------- Sidebar controls ----------
chatiness = st.sidebar.slider("Chatiness", min_value=1, max_value=10, value=7)
max_tokens_slider = st.sidebar.slider("Max tokens (slider)", min_value=1, max_value=10, value=3)
temperature = 0.4 + (chatiness - 1) * 0.06
max_tokens = 30 + (max_tokens_slider - 1) * 30

selected_domain = st.sidebar.radio(
    "Choose your round:",
    ["DSA", "AI", "SOFTWARE-ENGINEER-ROUND", "SYSTEM-DESIGN", "HR-ROUND"],
)

# ---------- NEW: Programming language selector (for DSA solutions) ----------
selected_prog_lang = st.sidebar.selectbox(
    "Preferred programming language (for DSA solutions):",
    ["Any", "C++", "Java", "Python"]
)

# ---------- Helper: detect programming language from solution text ----------
def detect_code_language(text: str) -> str:
    """
    Return one of: "C++", "Java", "Python", or "Other".
    Uses explicit markers (Solution (in ...), code fences) then heuristics.
    """
    if not text:
        return "Other"
    t = text.lower()

    # explicit markers
    if re.search(r"solution\s*\(in\s*c\+\+\)|solution\s*\(in\s*cpp\)|```cpp", t):
        return "C++"
    if re.search(r"solution\s*\(in\s*c\+\+\)|\<\?php", t):  # keep cautious
        return "C++"
    if re.search(r"solution\s*\(in\s*java\)|```java|system\.out\.println", t):
        return "Java"
    if re.search(r"solution\s*\(in\s*python\)|solution\s*\(in\s*py\)|```py|```python|def\s+\w+\(|print\(", t):
        return "Python"

    # heuristics
    # C++ heur
    if re.search(r"\b#include\s+<|std::|cout\b|cin\b|->\b|long long\b", text):
        return "C++"
    # Java heur
    if re.search(r"\bpublic static void main\b|System\.out\.println|class\s+\w+\s*{", text):
        return "Java"
    # Python heur
    if re.search(r"^\s*def\s+\w+\(|:\s*#\s*python", text, flags=re.M) or re.search(r"\bself\b", text):
        return "Python"

    return "Other"

# ---------- Document loader ----------
@st.cache_data
def load_documents_for_domain(domain: str, preferred_lang: str = "Any", limit: int = 20000) -> Tuple[List[Document], List[str]]:
    """
    Returns (docs_for_vectorstore, question_texts).
    For DSA: filters samples by detected solution programming language if preferred_lang != "Any".
    """
    domain = domain.upper()
    docs_for_vectorstore: List[Document] = []
    question_texts: List[str] = []

    if domain == "DSA":
        stream_data = load_dataset("sriniidhi/coding-dataset", split="train", streaming=True)
        data = list(stream_data.take(limit))
        for row in data:
            prompt = (row.get("prompt") or "").strip()
            response = (row.get("response") or "").strip()
            if not prompt:
                continue

            # detect language of the response
            lang = detect_code_language(response)

            # If user selected Any, accept all; otherwise filter by preferred programming language
            if preferred_lang != "Any":
                # map user choice to detector labels
                desired = preferred_lang
                if desired == "C++":
                    match = (lang == "C++")
                elif desired == "Java":
                    match = (lang == "Java")
                elif desired == "Python":
                    match = (lang == "Python")
                else:
                    match = False
                if not match:
                    continue  # skip this sample (solution in different language)

            # keep question text for UI (prompt only), and vectorstore doc contains both prompt+solution
            question_texts.append(prompt)
            docs_for_vectorstore.append(Document(page_content=f"QUESTION:\n{prompt}\n\nSOLUTION:\n{response}"))

    elif domain == "AI":
        ds = load_dataset("manasuma/ml_interview_qa")["train"]
        for row in ds:
            q = (row.get("Questions") or "").strip()
            a = (row.get("Answers") or "").strip()
            if q:
                question_texts.append(q)
                docs_for_vectorstore.append(Document(page_content=f"{q}\n\nAnswer:\n{a}"))

    elif domain == "SOFTWARE-ENGINEER-ROUND":
        ds = load_dataset("K-areem/AI-Interview-Questions")["train"]
        for row in ds:
            text = (row.get("text") or "").strip()
            cleaned = text.replace("<s>", "").replace("</s>", "").replace("[INST]", "").replace("[/INST]", "").strip()
            question_texts.append(cleaned)
            docs_for_vectorstore.append(Document(page_content=cleaned))

    elif domain == "SYSTEM-DESIGN" or domain == "SYSTEM-DESGIN":
        ds = load_dataset("SaffalPoosh/system_design")["train"]
        for row in ds:
            instruction = (row.get("instruction") or "").strip()
            output = (row.get("output") or "").strip()
            if instruction:
                question_texts.append(instruction)
                docs_for_vectorstore.append(Document(page_content=f"{instruction}\n\n{output}"))

    else:  # HR round
        stream_data = load_dataset("Ankshi/hr-interview-dataset", split="train", streaming=True)
        dataset = list(stream_data.take(limit))
        for row in dataset:
            q = (row.get("question") or "").strip()
            ideal = (row.get("ideal_answer") or "").strip()
            if q:
                question_texts.append(q)
                docs_for_vectorstore.append(Document(page_content=f"{q}\n\nIdeal Answer:\n{ideal}"))

    # fallback: if no question_texts created (rare), create from docs_for_vectorstore
    if not question_texts and docs_for_vectorstore:
        question_texts = [d.page_content for d in docs_for_vectorstore]

    return docs_for_vectorstore, question_texts

# ---------- Vectorstore builder ----------
def create_vectorstore(documents: List[Document]) -> Tuple:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever, chunks

def sample_questions(question_texts: List[str], n: int):
    copy = question_texts.copy()
    random.shuffle(copy)
    return copy[:n]

# ---------- Session state defaults ----------
QUESTIONS_PER_SESSION = 10
default_keys = {
    "current_domain": selected_domain,
    "intro_done": False,
    "question_number": 1,
    "questions": [],
    "current_question": "",
    "completed": False,
    "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    "qa_chain": None,
    "grading_chain": None,
    # store the selected programming language so we can trigger rebuilds when it changes
    "selected_prog_lang": selected_prog_lang
}

for key, val in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------- Domain switcher and chain builder ----------
def switch_domain_and_build():
    # rebuild if domain changed OR questions empty OR preferred programming language changed
    need_rebuild = (
        (selected_domain != st.session_state.current_domain)
        or (not st.session_state.questions)
        or (st.session_state.get("selected_prog_lang") != selected_prog_lang)
    )
    if not need_rebuild:
        return

    st.session_state.current_domain = selected_domain
    st.session_state.intro_done = False
    st.session_state.question_number = 1
    st.session_state.completed = False
    st.session_state.memory.clear()
    st.session_state.selected_prog_lang = selected_prog_lang

    # pass preferred programming language into loader so it can filter DSA samples
    docs_for_vectorstore, question_texts = load_documents_for_domain(selected_domain, preferred_lang=selected_prog_lang, limit=2000)
    retriever, _ = create_vectorstore(docs_for_vectorstore)
    st.session_state.questions = sample_questions(question_texts, QUESTIONS_PER_SESSION)
    st.session_state.current_question = st.session_state.questions[0] if st.session_state.questions else ""

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful AI interview assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Provide a clear and complete answer.
""",
    )

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model_name="llama3-70b-8192",
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key,
            streaming=True,
        ),
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    grading_prompt = PromptTemplate(
        input_variables=["question", "answer", "ideal_answer"],
        template="""
You are an evaluator grading a user's interview answer.

Question: {question}
User's Answer: {answer}
Ideal Answer: {ideal_answer}

First, check correctness:
- If the user's answer matches or is mostly correct, mark "Correct".
- Else, mark "Incorrect".

Then give:
- Feedback: 1–2 lines explaining why it's correct or incorrect.
- Encouragement: 1 motivational line to help the user improve.

Return ONLY in JSON:
{{"Correctness": "...", "Feedback": "...", "Encouragement": "..."}}
""",
    )

    st.session_state.grading_chain = LLMChain(
        llm=ChatOpenAI(
            model_name="llama3-70b-8192",  
            temperature=temperature,
            max_tokens=200,
            openai_api_key=api_key,
            streaming=False,
        ),
        prompt=grading_prompt,
    )

switch_domain_and_build()

# ---------- UI ----------
if not st.session_state.intro_done:
    with st.chat_message("assistant"):
        st.markdown(f"👋 Welcome to your **{selected_domain}** interview practice!")
        st.markdown("Please start with a short introduction about yourself.")

user_input = st.chat_input("your response")

if user_input:
    st.chat_message("user").markdown(user_input)

    if not st.session_state.intro_done:
        with st.chat_message("assistant"):
            st.markdown("✅ Thanks for the introduction! Let's begin with the questions.")
        st.session_state.intro_done = True
        if st.session_state.questions:
            st.session_state.current_question = st.session_state.questions[0]

    elif not st.session_state.completed:
        question = st.session_state.current_question or "No question loaded."

        with st.chat_message("assistant"):
            st.markdown(f"**Question {st.session_state.question_number}/{QUESTIONS_PER_SESSION}:**\n{question}")

        with st.chat_message("assistant"):
            response_container = st.empty()
            stream = StreamHandler(response_container)
            ideal_answer = ""
            try:
                result = st.session_state.qa_chain.invoke({"question": question}, callbacks=[stream])
                if isinstance(result, dict):
                    ideal_answer = result.get("answer") or result.get("result") or ""
                elif isinstance(result, str):
                    ideal_answer = result
                else:
                    ideal_answer = str(result)
            except Exception as e:
                st.error(f"QA chain failed: {e}")
                ideal_answer = ""

        # Show ideal answer before grading
        with st.chat_message("assistant"):
            st.markdown(f"**Ideal Answer:**\n{ideal_answer}")

        try:
            grading_raw = st.session_state.grading_chain.run({
                "question": question,
                "answer": user_input,
                "ideal_answer": ideal_answer
            })
            parsed = None
            try:
                parsed = json.loads(grading_raw)
            except Exception:
                try:
                    start = grading_raw.find("{")
                    end = grading_raw.rfind("}") + 1
                    if start != -1 and end != -1:
                        parsed = json.loads(grading_raw[start:end])
                except Exception:
                    parsed = None

            if parsed and isinstance(parsed, dict):
                with st.chat_message("assistant"):
                    st.markdown(f"**Correctness:** {parsed.get('Correctness', 'Unknown')}")
                    st.markdown(f"**Feedback:** {parsed.get('Feedback', 'No feedback.')}")
                    st.markdown(f"**Encouragement:** {parsed.get('Encouragement', 'Keep going!')}")
            else:
                with st.chat_message("assistant"):
                    st.markdown("Unable to parse grader output. Raw output:")
                    st.text(grading_raw)

        except Exception as e:
            st.error(f"Grading failed: {e}")

        st.session_state.question_number += 1
        if st.session_state.question_number > QUESTIONS_PER_SESSION:
            st.session_state.completed = True
        else:
            idx = st.session_state.question_number - 1
            if idx < len(st.session_state.questions):
                st.session_state.current_question = st.session_state.questions[idx]
            else:
                st.session_state.completed = True

if st.session_state.intro_done and not st.session_state.completed and st.session_state.current_question:
    st.chat_message("assistant").markdown(
        f"**Question {st.session_state.question_number}/{QUESTIONS_PER_SESSION}**:\n{st.session_state.current_question}"
    )

if st.session_state.completed:
    with st.chat_message("assistant"):
        st.success("🎉 Interview Complete! You’ve done a great job. Keep practicing and good luck with your real interviews!")

