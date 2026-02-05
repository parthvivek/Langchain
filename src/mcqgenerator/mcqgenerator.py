from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os
import json
import pandas as pd
import warnings
import logging
from datetime import datetime
import time

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING SETUP
# ============================================================================

os.makedirs("logs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

log_filename = f"logs/mcq_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set to True to skip review/fix and save API quota
SIMPLE_MODE = False

# Maximum retry attempts for review/fix cycle
MAX_RETRIES = 3

logger.info("="*70)
logger.info("INITIALIZING MCQ GENERATOR PIPELINE")
logger.info("="*70)
logger.info(f"Simple Mode: {SIMPLE_MODE}")
logger.info(f"Max Retries: {MAX_RETRIES}")

# ============================================================================
# ENVIRONMENT & MODEL SETUP
# ============================================================================

load_dotenv(override=True)
key = os.getenv("API_KEY")

if not key:
    logger.error("❌ API_KEY not found in environment variables")
    raise ValueError("API_KEY not found in .env file")

logger.info("✅ Environment variables loaded")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=key
)

logger.info("✅ LLM initialized: gemini-2.5-flash")

# ============================================================================
# TEST API QUOTA
# ============================================================================

logger.info("\n" + "-"*70)
logger.info("TESTING API CONNECTIVITY")
logger.info("-"*70)

try:
    test_response = llm.invoke("Hi")
    logger.info("✅ API connection successful")
except Exception as e:
    if "429" in str(e) or "quota" in str(e).lower():
        logger.critical("🚨 API QUOTA EXCEEDED!")
        logger.info("❌ You've hit the free tier limit (20 requests/day)")
        logger.info("💡 Solutions:")
        logger.info("   1. Wait ~24 hours for quota reset")
        logger.info("   2. Upgrade to paid plan: https://ai.google.dev/pricing")
        logger.info("   3. Use a different API key")
        logger.info("   4. Set SIMPLE_MODE=True to skip review/fix")
        raise SystemExit("Quota exceeded - exiting")
    else:
        logger.error(f"❌ API test failed: {e}")
        raise

# ============================================================================
# USER INPUTS
# ============================================================================

logger.info("\n" + "-"*70)
logger.info("COLLECTING USER INPUTS")
logger.info("-"*70)

try:
    number = int(input("Enter number of options per question: "))
    tone = input("Enter difficulty level (easy, medium, hard): ").strip().lower()
    no_of = int(input("Enter number of questions to generate: "))
    
    logger.info(f"✅ User inputs collected:")
    logger.info(f"   - Options per question: {number}")
    logger.info(f"   - Difficulty: {tone}")
    logger.info(f"   - Number of questions: {no_of}")
    
except ValueError as e:
    logger.error(f"❌ Invalid input: {e}")
    raise

# ============================================================================
# LOAD INPUT TEXT
# ============================================================================

logger.info("\n" + "-"*70)
logger.info("LOADING INPUT TEXT")
logger.info("-"*70)

try:
    input_text = TextLoader("test_v1.txt", encoding="utf-8").load()[0].page_content
    logger.info(f"✅ Text loaded: {len(input_text)} characters")
    logger.info(f"   Preview: {input_text[:100]}...")
except Exception as e:
    logger.error(f"❌ Failed to load text file: {e}")
    raise

# ============================================================================
# LOAD JSON TEMPLATE
# ============================================================================

logger.info("\n" + "-"*70)
logger.info("LOADING JSON TEMPLATE")
logger.info("-"*70)

try:
    with open(r"C:\Users\HP\Documents\Projects\Generative AI\Langchain\src\mcqgenerator\response.js", "r") as file:
        response_json = json.load(file)
    logger.info("✅ JSON template loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load JSON template: {e}")
    raise

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

logger.info("\n" + "-"*70)
logger.info("SETTING UP PROMPT CHAINS")
logger.info("-"*70)

# QUIZ GENERATION PROMPT
quiz_template = """
You are an expert educational content generator.

TASK:
Generate Multiple Choice Questions (MCQs) from the given content.

INPUT DATA:
Text: {text}
Subject: {subject}

INSTRUCTIONS:
- Generate exactly {no_of} MCQs
- Each MCQ must have exactly {number} options
- Only ONE option must be correct
- Difficulty level must be: {tone}
- Questions must NOT repeat
- Questions must NOT be semantically similar
- Questions must be concept-based, not copy-paste lines
- Questions must be clearly framed
- Options must be realistic and confusing (but fair)
- Avoid ambiguous questions
- Avoid multiple correct answers
- Do NOT include explanations
- Do NOT include extra text outside JSON

OUTPUT FORMAT:
Strictly follow the JSON structure below.

RESPONSE_JSON:
{response_json}

IMPORTANT RULES:
- Output must be valid JSON
- No markdown
- No commentary
- No extra keys
- No missing fields
- No trailing commas
- No text outside JSON
"""

quiz_prompt = PromptTemplate(
    input_variables=["text", "subject", "no_of", "number", "tone", "response_json"],
    template=quiz_template
)

quiz_chain = quiz_prompt | llm
logger.info("✅ Quiz generation chain created")

# REVIEW PROMPT
review_template = """
You are an expert examiner and subject matter validator.

You will receive:
1) Original source text
2) Generated MCQs in JSON format
3) Difficulty tone

Your task:
Validate each MCQ for:
- Conceptual correctness
- Factual correctness
- One and only one correct answer
- Answer correctness
- No ambiguity
- No hallucination
- Proper difficulty as per tone
- No repeated or similar options
- Logical clarity
- Relevance to the source text

### OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "overall_score": "0-100",
  "summary": "short quality summary",
  "verdict": "PASS / FAIL",
  "errors_count": 0,
  "question_reviews": [
    {{
      "question_id": 1,
      "status": "PASS / FAIL",
      "issues": [
        "issue1",
        "issue2"
      ],
      "difficulty_match": true,
      "answer_correct": true,
      "logic_valid": true,
      "hallucination": false,
      "improvement_suggestion": "text"
    }}
  ]
}}

Rules:
- Be strict
- Do not rewrite MCQs
- Do not explain outside JSON
- No markdown
- No comments

SOURCE TEXT:
{source_text}

MCQ JSON:
{mcq_json}

TONE:
{tone}
"""

review_prompt = PromptTemplate(
    input_variables=["source_text", "mcq_json", "tone"],
    template=review_template
)

review_chain = review_prompt | llm
logger.info("✅ Review chain created")

# FIX PROMPT
fix_template = """
You are an expert AI question corrector.

You will receive:
1) Original source text
2) Generated MCQs JSON
3) Review report JSON with errors

Your task:
- Fix ONLY the incorrect questions
- Keep correct questions unchanged
- Ensure:
  - Conceptual correctness
  - Factual correctness
  - One correct answer
  - No hallucinations
  - Proper difficulty
  - JSON format consistency
  - Same structure as input MCQ JSON

### OUTPUT FORMAT:
Return ONLY corrected MCQ JSON (no explanation, no markdown).

SOURCE TEXT:
{text}

MCQ JSON:
{mcq_json}

REVIEW JSON:
{review_json}
"""

fix_prompt = PromptTemplate(
    input_variables=["text", "mcq_json", "review_json"],
    template=fix_template
)

fix_chain = fix_prompt | llm
logger.info("✅ Fix chain created")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_json_parse(text):
    """Safely parse JSON from LLM response with logging"""
    logger.debug("Attempting to parse JSON response")
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
        logger.debug("Removed ```json markdown")
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    # Find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start == -1 or end == 0:
        logger.error("No JSON object found in response")
        logger.debug(f"Raw response preview: {text[:200]}")
        raise ValueError("No JSON object found in response")
    
    clean = text[start:end]
    
    try:
        parsed = json.loads(clean)
        logger.debug("✅ JSON parsed successfully")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.debug(f"Failed JSON preview: {clean[:200]}")
        raise


def is_quota_error(error):
    """Check if error is a quota/rate limit error"""
    error_str = str(error).lower()
    return "429" in error_str or "quota" in error_str or "rate limit" in error_str


def create_default_review():
    """Create a default review object when review fails"""
    return {
        "overall_score": "N/A",
        "summary": "Review not performed",
        "verdict": "UNKNOWN",
        "errors_count": 0,
        "question_reviews": []
    }

# ============================================================================
# SIMPLE GENERATION (NO REVIEW/FIX)
# ============================================================================

def simple_quiz_generation(input_text, number, tone, no_of, response_json):
    """
    Simple MCQ generation without review/fix (saves API quota)
    """
    
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("\n" + "="*70)
    logger.info(f"🚀 SIMPLE MODE | Session ID: {session_id}")
    logger.info("="*70)
    logger.info("⚠️ Review and fix steps are SKIPPED to save API quota")
    
    start_time = datetime.now()
    
    try:
        logger.info("\n📝 Generating MCQs...")
        quiz_response = quiz_chain.invoke({
            "text": input_text,
            "subject": "Artificial Intelligence",
            "number": number,
            "tone": tone,
            "no_of": no_of,
            "response_json": json.dumps(response_json)
        })
        
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ MCQs generated in {generation_time:.2f}s")
        
        current_mcq = quiz_response.content.strip()
        
        return {
            "status": "SIMPLE_MODE",
            "mcq": json.loads(current_mcq),
            "review": {
                "overall_score": "N/A",
                "summary": "Review skipped in SIMPLE_MODE",
                "verdict": "NOT_REVIEWED",
                "errors_count": 0,
                "question_reviews": []
            },
            "attempts": 1,
            "session_id": session_id,
            "total_time": generation_time
        }
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}", exc_info=True)
        raise

# ============================================================================
# SELF-CORRECTING PIPELINE
# ============================================================================

def self_correcting_quiz_pipeline(input_text, number, tone, no_of, response_json):
    """
    Self-correcting quiz generation with review and fix loops
    """
    
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("\n" + "="*70)
    logger.info(f"🚀 STARTING FULL PIPELINE | Session ID: {session_id}")
    logger.info("="*70)
    logger.info(f"Parameters: questions={no_of}, options={number}, tone={tone}")
    
    # Step 1: Generate Quiz
    logger.info("\n📝 Step 1: Generating MCQs...")
    start_time = datetime.now()
    
    try:
        quiz_response = quiz_chain.invoke({
            "text": input_text,
            "subject": "Artificial Intelligence",
            "number": number,
            "tone": tone,
            "no_of": no_of,
            "response_json": json.dumps(response_json)
        })
        
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ MCQs generated in {generation_time:.2f}s")
        
        current_mcq = quiz_response.content.strip()
        logger.debug(f"Generated MCQ preview: {current_mcq[:200]}...")
        
    except Exception as e:
        if is_quota_error(e):
            logger.critical("🚨 API QUOTA EXCEEDED during generation!")
            logger.info("💡 Set SIMPLE_MODE=False or wait for quota reset")
        logger.error(f"❌ MCQ generation failed: {e}", exc_info=True)
        raise

    # Initialize review_result
    review_result = None
    
    # Review and Fix Loop
    for attempt in range(MAX_RETRIES):
        logger.info("\n" + "-"*70)
        logger.info(f"🔍 Review Attempt {attempt + 1}/{MAX_RETRIES}")
        logger.info("-"*70)

        # Step 2: Review
        logger.info("⏳ Reviewing MCQs...")
        review_start = datetime.now()
        
        try:
            review_response = review_chain.invoke({
                "source_text": input_text,
                "mcq_json": current_mcq,
                "tone": tone
            })

            review_time = (datetime.now() - review_start).total_seconds()
            logger.info(f"Review completed in {review_time:.2f}s")
            
            review_raw = review_response.content.strip()
            review_result = safe_json_parse(review_raw)
            
        except Exception as e:
            logger.error(f"❌ Review attempt {attempt + 1} failed: {e}", exc_info=True)
            
            if is_quota_error(e):
                logger.critical("🚨 API QUOTA EXCEEDED - Cannot proceed with review")
                logger.info("💡 Returning generated MCQs without review")
                
                total_time = (datetime.now() - start_time).total_seconds()
                return {
                    "status": "QUOTA_EXCEEDED",
                    "mcq": json.loads(current_mcq),
                    "review": create_default_review(),
                    "attempts": attempt + 1,
                    "session_id": session_id,
                    "total_time": total_time,
                    "error": "API quota exceeded during review"
                }
            
            continue

        # Only proceed if review succeeded
        if review_result is None:
            continue

        verdict = review_result.get('verdict', 'UNKNOWN')
        score = review_result.get('overall_score', 'N/A')
        errors = review_result.get('errors_count', 'N/A')
        
        logger.info(f"📊 Verdict: {verdict}")
        logger.info(f"📊 Score: {score}")
        logger.info(f"📊 Errors: {errors}")
        logger.info(f"📊 Summary: {review_result.get('summary', 'N/A')}")

        # Log question-level issues
        if 'question_reviews' in review_result:
            failed_questions = [q for q in review_result['question_reviews'] if q.get('status') == 'FAIL']
            if failed_questions:
                logger.warning(f"⚠️ Failed questions: {len(failed_questions)}/{len(review_result['question_reviews'])}")
                for q in failed_questions:
                    logger.warning(f"   Q{q.get('question_id')}: {', '.join(q.get('issues', []))}")

        # Step 3: Check Verdict
        if verdict == "PASS":
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info("\n" + "="*70)
            logger.info(f"✅ SUCCESS after {attempt + 1} attempt(s) in {total_time:.2f}s")
            logger.info("="*70)
            
            return {
                "status": "SUCCESS",
                "mcq": json.loads(current_mcq),
                "review": review_result,
                "attempts": attempt + 1,
                "session_id": session_id,
                "total_time": total_time
            }

        # Step 4: Fix if FAIL
        logger.info(f"\n🛠 Fixing incorrect questions (Attempt {attempt + 1})...")
        fix_start = datetime.now()
        
        try:
            fixed_response = fix_chain.invoke({
                "text": input_text,
                "mcq_json": current_mcq,
                "review_json": json.dumps(review_result)
            })

            fix_time = (datetime.now() - fix_start).total_seconds()
            current_mcq = fixed_response.content.strip()
            logger.info(f"✅ MCQs corrected in {fix_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Fix attempt {attempt + 1} failed: {e}", exc_info=True)
            
            if is_quota_error(e):
                logger.critical("🚨 API QUOTA EXCEEDED - Cannot fix MCQs")
                total_time = (datetime.now() - start_time).total_seconds()
                return {
                    "status": "QUOTA_EXCEEDED",
                    "mcq": json.loads(current_mcq),
                    "review": review_result if review_result else create_default_review(),
                    "attempts": attempt + 1,
                    "session_id": session_id,
                    "total_time": total_time,
                    "error": "API quota exceeded during fix"
                }
            
            continue

    # If still failing after retries
    total_time = (datetime.now() - start_time).total_seconds()
    logger.warning("\n" + "="*70)
    logger.warning(f"⚠️ FAILED after {MAX_RETRIES} attempts in {total_time:.2f}s")
    logger.warning("="*70)
    
    return {
        "status": "FAILED_AFTER_RETRIES",
        "mcq": json.loads(current_mcq),
        "review": review_result if review_result else create_default_review(),
        "attempts": MAX_RETRIES,
        "session_id": session_id,
        "total_time": total_time
    }

# ============================================================================
# RUN PIPELINE
# ============================================================================

try:
    if SIMPLE_MODE:
        result = simple_quiz_generation(
            input_text=input_text,
            number=number,
            tone=tone,
            no_of=no_of,
            response_json=response_json
        )
    else:
        result = self_correcting_quiz_pipeline(
            input_text=input_text,
            number=number,
            tone=tone,
            no_of=no_of,
            response_json=response_json
        )
    
    logger.info("\n" + "="*70)
    logger.info("📋 FINAL RESULTS")
    logger.info("="*70)
    logger.info(f"Status: {result['status']}")
    logger.info(f"Attempts: {result['attempts']}")
    logger.info(f"Total Time: {result.get('total_time', 0):.2f}s")
    logger.info(f"Session ID: {result['session_id']}")
    
    if 'error' in result:
        logger.warning(f"⚠️ Error: {result['error']}")
    
except Exception as e:
    logger.critical(f"💥 PIPELINE CRASHED: {e}", exc_info=True)
    raise

# ============================================================================
# CREATE DATAFRAME
# ============================================================================

logger.info("\n" + "-"*70)
logger.info("📊 CREATING DATAFRAME")
logger.info("-"*70)

try:
    mcq_data = result["mcq"]

    # Normalization
    if isinstance(mcq_data, dict) and "mcqs" in mcq_data:
        mcqs = mcq_data["mcqs"]
    elif isinstance(mcq_data, dict) and "mcq" in mcq_data:
        mcqs = mcq_data["mcq"]
    elif isinstance(mcq_data, dict):
        mcqs = list(mcq_data.values())
    elif isinstance(mcq_data, list):
        mcqs = mcq_data
    else:
        raise ValueError("❌ Unknown MCQ JSON structure")

    logger.info(f"✅ Normalized {len(mcqs)} MCQs")

    # Build DataFrame
    rows = []
    for idx, mcq in enumerate(mcqs, start=1):
        question = mcq.get("question", "")
        options = mcq.get("options", {})

        if isinstance(options, list):
            opt_text = " | ".join([f"{opt.get('label', '')}: {opt.get('text', '')}" for opt in options])
            answer = next((opt.get('label', '') for opt in options if opt.get("is_correct", False)), "N/A")
        elif isinstance(options, dict):
            opt_text = " | ".join([f"{k}: {v}" for k, v in options.items()])
            answer = mcq.get("correct_answer", "N/A")
        else:
            opt_text = str(options)
            answer = "N/A"

        rows.append({
            "No": idx,
            "Question": question,
            "Options": opt_text,
            "Answer": answer
        })

    df_mcq = pd.DataFrame(rows)
    logger.info(f"✅ DataFrame created with {len(df_mcq)} rows")

except Exception as e:
    logger.error(f"❌ DataFrame creation failed: {e}", exc_info=True)
    raise

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "="*70)
print("📊 MCQ DATAFRAME")
print("="*70)
print(df_mcq.to_string(index=False))

print("\n" + "="*70)
print("📋 REVIEW SUMMARY")
print("="*70)
print(f"Status: {result['status']}")
print(f"Overall Score: {result['review'].get('overall_score', 'N/A')}")
print(f"Verdict: {result['review'].get('verdict', 'N/A')}")
print(f"Errors Count: {result['review'].get('errors_count', 'N/A')}")
print(f"Summary: {result['review'].get('summary', 'N/A')}")

# Display question-level issues
if 'question_reviews' in result['review']:
    failed = [q for q in result['review']['question_reviews'] if q.get('status') == 'FAIL' or q.get('issues')]
    if failed:
        print("\n" + "="*70)
        print("⚠️ QUESTION-LEVEL ISSUES")
        print("="*70)
        for q_review in failed:
            print(f"\nQuestion {q_review.get('question_id', 'N/A')}:")
            print(f"  Status: {q_review.get('status', 'N/A')}")
            if q_review.get('issues'):
                print(f"  Issues: {', '.join(q_review['issues'])}")
            if q_review.get('improvement_suggestion'):
                print(f"  Suggestion: {q_review['improvement_suggestion']}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

logger.info("\n" + "-"*70)
logger.info("💾 SAVING OUTPUTS")
logger.info("-"*70)

try:
    session_id = result['session_id']
    
    # Save CSV
    csv_path = f"outputs/mcq_{session_id}.csv"
    df_mcq.to_csv(csv_path, index=False)
    logger.info(f"✅ CSV saved: {csv_path}")
    
    # Save Review
    review_path = f"outputs/review_{session_id}.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(result['review'], f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Review saved: {review_path}")
    
    # Save MCQs
    mcq_path = f"outputs/mcq_final_{session_id}.json"
    with open(mcq_path, "w", encoding="utf-8") as f:
        json.dump(result['mcq'], f, indent=2, ensure_ascii=False)
    logger.info(f"✅ MCQs saved: {mcq_path}")
    
except Exception as e:
    logger.error(f"❌ File save failed: {e}", exc_info=True)

# ============================================================================
# COMPLETION
# ============================================================================

logger.info("\n" + "="*70)
logger.info(f"🎉 PIPELINE COMPLETED")
logger.info(f"📄 Log file: {log_filename}")
logger.info("="*70 + "\n")

print("\n" + "="*70)
print(f"✅ All outputs saved successfully!")
print(f"📄 Log file: {log_filename}")
print(f"📁 Output files: outputs/")
print("="*70)