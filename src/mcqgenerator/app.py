import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# ============================
# CONFIG
# ============================
load_dotenv(override=True)
API_KEY = os.getenv("API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=API_KEY
)

st.set_page_config(page_title="AI MCQ Generator", layout="wide")

# ============================
# UI HEADER
# ============================
st.title("🧠 AI MCQ Generator Platform")
st.subheader("Self-Correcting MCQ System | Evaluation Engine | Auto-Fix Pipeline")
st.markdown("---")

# ============================
# SIDEBAR CONTROLS
# ============================
st.sidebar.title("⚙️ Configuration")

no_of = st.sidebar.number_input("Number of Questions", min_value=1, max_value=50, value=5)
number = st.sidebar.number_input("Options per Question", min_value=2, max_value=6, value=4)
tone = st.sidebar.selectbox("Difficulty Level", ["easy", "medium", "hard"])
uploaded_file = st.sidebar.file_uploader("Upload Text File", type=["txt"])

# ============================
# PROMPTS
# ============================

# ---- Generation Prompt
quiz_template = """
Generate MCQs from text.

Rules:
- {no_of} questions
- {number} options each
- one correct answer
- difficulty: {tone}
- JSON only

FORMAT:
{{
  "mcqs": [
    {{
      "question": "",
      "options": [
        {{"label":"A","text":"","is_correct":false}}
      ]
    }}
  ]
}}

TEXT:
{text}
"""

quiz_prompt = PromptTemplate(
    input_variables=["text", "no_of", "number", "tone"],
    template=quiz_template
)

quiz_chain = quiz_prompt | llm

# ---- Review Prompt
review_template = """
You are an AI MCQ evaluator.

Evaluate each MCQ on:
- conceptual correctness
- single correct answer
- clarity
- ambiguity
- option quality
- difficulty match
- language quality

Return JSON strictly in this format:

{{
  "reviews": [
    {{
      "question": "",
      "verdict": "PASS/FAIL",
      "reason": ""
    }}
  ]
}}

TEXT:
{text}

MCQs:
{mcq_json}
"""

review_prompt = PromptTemplate(
    input_variables=["text", "mcq_json"],
    template=review_template
)

review_chain = review_prompt | llm

# ---- Fix Prompt
fix_template = """
You are an AI MCQ correction engine.

Fix the FAILED MCQs only.

Rules:
- Correct conceptual errors
- Remove ambiguity
- Ensure one correct answer
- Improve clarity
- Match difficulty
- Improve options quality

Return JSON strictly in this format:

{{
  "mcqs": [
    {{
      "question": "",
      "options": [
        {{"label":"A","text":"","is_correct":false}}
      ]
    }}
  ]
}}

FAILED MCQs:
{failed_mcqs}
"""

fix_prompt = PromptTemplate(
    input_variables=["failed_mcqs"],
    template=fix_template
)

fix_chain = fix_prompt | llm

# ============================
# FUNCTIONS
# ============================
def safe_json(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])

# ============================
# MAIN PANEL
# ============================
st.markdown("### 📥 Input")

input_text = ""
if uploaded_file:
    input_text = uploaded_file.read().decode("utf-8")
    st.success("Text file uploaded successfully")

st.text_area("Source Text", value=input_text, height=200)

# ============================
# GENERATE PIPELINE
# ============================
if st.button("🚀 Generate MCQs"):
    if not input_text.strip():
        st.error("Please upload or paste text")
    else:
        with st.spinner("Step 1: Generating MCQs..."):
            response = quiz_chain.invoke({
                "text": input_text,
                "no_of": no_of,
                "number": number,
                "tone": tone
            })
            mcq_json = safe_json(response.content)

        st.success("MCQs Generated ✅")

        # ============================
        # REVIEW
        # ============================
        with st.spinner("Step 2: Evaluating MCQs..."):
            review_response = review_chain.invoke({
                "text": input_text,
                "mcq_json": json.dumps(mcq_json, indent=2)
            })
            review_json = safe_json(review_response.content)

        st.success("Evaluation Completed ✅")

        # ============================
        # DISPLAY REVIEW
        # ============================
        st.markdown("---")
        st.markdown("## 🧪 Evaluation Report")

        failed_questions = []
        passed_questions = []

        for r in review_json["reviews"]:
            if r["verdict"] == "FAIL":
                st.error(f"❌ {r['question']} — {r['reason']}")
                failed_questions.append(r["question"])
            else:
                st.success(f"✅ {r['question']}")

        # ============================
        # EXTRACT FAILED MCQs
        # ============================
        failed_mcqs = []
        for mcq in mcq_json["mcqs"]:
            if mcq["question"] in failed_questions:
                failed_mcqs.append(mcq)
            else:
                passed_questions.append(mcq)

        # ============================
        # FIX FAILED MCQs
        # ============================
        fixed_mcqs = []

        if failed_mcqs:
            with st.spinner("Step 3: Auto-fixing failed MCQs..."):
                fix_response = fix_chain.invoke({
                    "failed_mcqs": json.dumps(failed_mcqs, indent=2)
                })
                fixed_json = safe_json(fix_response.content)
                fixed_mcqs = fixed_json["mcqs"]

            st.success("Auto-fix completed ✅")

        # ============================
        # FINAL MERGE
        # ============================
        final_mcqs = passed_questions + fixed_mcqs

        # ============================
        # DISPLAY FINAL MCQs
        # ============================
        st.markdown("---")
        st.markdown("## ✅ Final Validated MCQs")

        rows = []
        for i, mcq in enumerate(final_mcqs, 1):
            st.markdown(f"### Q{i}. {mcq['question']}")
            for opt in mcq["options"]:
                if opt.get("is_correct"):
                    st.markdown(f"✅ **{opt['label']}. {opt['text']}**")
                else:
                    st.markdown(f"{opt['label']}. {opt['text']}")

            rows.append({
                "Question": mcq["question"],
                "Options": " | ".join([f"{o['label']}: {o['text']}" for o in mcq["options"]]),
                "Answer": next(o["label"] for o in mcq["options"] if o["is_correct"])
            })

        df = pd.DataFrame(rows)

        # ============================
        # TABLE
        # ============================
        st.markdown("---")
        st.markdown("## 📊 Final MCQ Dataset")
        st.dataframe(df, use_container_width=True)

        # ============================
        # DOWNLOADS
        # ============================
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_data = df.to_csv(index=False).encode('utf-8')
        json_data = json.dumps({"mcqs": final_mcqs}, indent=2).encode('utf-8')

        st.download_button("⬇️ Download Clean CSV", csv_data, file_name=f"mcq_clean_{session_id}.csv")
        st.download_button("⬇️ Download Clean JSON", json_data, file_name=f"mcq_clean_{session_id}.json")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("### 🧠 AI MCQ Platform")
st.markdown("Self-Correcting AI | Evaluation Engine | Auto-Fix Pipeline | Production Architecture")
