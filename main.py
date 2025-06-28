from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
from supabase import create_client, Client
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import io
import os
from pydantic import BaseModel
import json
import asyncio
import logging
import re
import requests
import time
import tempfile
import dotenv
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.GEMINI_API_KEY = "AIzaSyBoUdOFtm6VgmUdzkiTM5bW67TJXc5zMk0"
        self.SUPABASE_URL = "https://xrahjhhjeamprikyjwyg.supabase.co"
        self.SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhyYWhqaGhqZWFtcHJpa3lqd3lnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMDU3ODgsImV4cCI6MjA2NDU4MTc4OH0.9PCS7wxdTuxx2PUMyH5nA2A-dVFxGV5FUXgV0ePqHEY"
        self.SUPABASE_BUCKET = "mcq-files"
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# EVAL_SUPABASE_KEY = os.getenv('EVAL_SUPABASE_KEY')
# EVAL_SUPABASE_URL = os.getenv('EVAL_SUPABASE_URL')
# MCQ_SUPABASE_BUCKET = os.getenv('MCQ_SUPABASE_BUCKET')
# OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
# SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET')
# SUPABASE_KEY = os.getenv('SUPABASE_KEY')
# SUPABASE_URL = os.getenv('SUPABASE_URL')
# TAVUS_API_KEY = os.getenv('TAVUS_API_KEY')
# TAVUS_BASE_URL = os.getenv('TAVUS_BASE_URL')
# MCQ_SUPABASE_URL = os.getenv('EVAL_SUPABASE_KEY')
# MCQ_SUPABASE_KEY = os.getenv('EVAL_SUPABASE_URL')
# Initialize FastAPI app
app = FastAPI(title="Combined EduTech API", version="1.0.0")
config = Config()
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://classroom-3ymf-pfcm0wuoa-azmaininqiads-projects.vercel.app",
        "https://classroom-3ymf.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Add any other origins you need
    ],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== Configuration ==================
# Evaluation API Config
EVAL_SUPABASE_URL = "https://xrahjhhjeamprikyjwyg.supabase.co"
EVAL_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhyYWhqaGhqZWFtcHJpa3lqd3lnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDkwMDU3ODgsImV4cCI6MjA2NDU4MTc4OH0.9PCS7wxdTuxx2PUMyH5nA2A-dVFxGV5FUXgV0ePqHEY"
GEMINI_API_KEY = "AIzaSyBoUdOFtm6VgmUdzkiTM5bW67TJXc5zMk0"

# MCQ Generator Config
MCQ_SUPABASE_URL = "https://mixwsmdaogjctmiwiogz.supabase.co"
MCQ_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1peHdzbWRhb2dqY3RtaXdpb2d6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk5ODcyNzksImV4cCI6MjA2NTU2MzI3OX0.6Mf_49GGiEPEyMdiadmjcDjLHo26M8GxPpMGmikFEAc"
MCQ_SUPABASE_BUCKET = "mcq-files"

# Tavus Config
TAVUS_API_KEY = "caa76efefe4e4a8caf2a94b05454b668"
TAVUS_BASE_URL = "https://tavusapi.com/v2"

# OpenRouter Config
OPENROUTER_API_KEY = "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"

# ================== Initialize Services ==================
# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
eval_model = genai.GenerativeModel('gemini-2.0-flash-exp')
mcq_model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Initialize Supabase clients
eval_supabase: Client = create_client(EVAL_SUPABASE_URL, EVAL_SUPABASE_KEY)
mcq_supabase: Client = create_client(MCQ_SUPABASE_URL, MCQ_SUPABASE_KEY)
supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
# ================== Pydantic Models ==================
# Evaluation Models
class EvaluationResult(BaseModel):
    id: str
    student_name: str
    total_marks: int
    obtained_marks: int
    percentage: float
    grade: str
    correct_answers: List[str]
    incorrect_answers: List[str]
    partial_credit_areas: List[str]
    strengths: List[str]
    areas_for_improvement: List[str]
    detailed_feedback: str
    timestamp: str
    evaluation_type: str

class EvaluationSummary(BaseModel):
    average_percentage: float
    grade_distribution: Dict[str, int]
    highest_score: float
    lowest_score: float

class SingleEvaluationResponse(BaseModel):
    success: bool
    result: Optional[EvaluationResult] = None
    message: str

class MultipleEvaluationResponse(BaseModel):
    success: bool
    results: Optional[List[EvaluationResult]] = None
    summary: Optional[EvaluationSummary] = None
    message: str
    total_students: Optional[int] = None

# Tavus Models
class TavusConversationRequest(BaseModel):
    conversation_name: str = "EduBot Video Chat"
    conversational_context: str = "You are EduBot, an educational AI assistant helping students learn."
    custom_greeting: str = "Hello! I'm EduBot, your AI learning companion."

class EndConversationRequest(BaseModel):
    conversation_id: str

# Course Generator Models
class CourseRequest(BaseModel):
    subject: str

class CourseResponse(BaseModel):
    toc: list
    content: dict
    headlines: dict

# ================== Helper Functions ==================
def get_file_mime_type(filename: str) -> str:
    """Get MIME type based on file extension."""
    extension = filename.lower().split('.')[-1]
    mime_types = {
        'pdf': 'application/pdf',
        'txt': 'text/plain',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png'
    }
    return mime_types.get(extension, 'application/octet-stream')

def calculate_grade(percentage: float) -> str:
    """Calculate letter grade based on percentage."""
    if percentage >= 90:
        return 'A'
    elif percentage >= 80:
        return 'B'
    elif percentage >= 70:
        return 'C'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'

def generate_markdown(prompt: str, model: str = "deepseek/deepseek-r1-0528-qwen3-8b:free") -> str:
    """Generate markdown content using OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "localhost",
        "X-Title": "Course Generator"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 8000,
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

def parse_course_content(markdown_content: str):
    """Parse generated course content into structured format."""
    lines = markdown_content.split('\n')
    toc = []
    content = {}
    headlines = {}
    current_topic = None
    current_content = []
    
    for line in lines:
        # Main topic (## 1. Topic Name)
        if line.startswith('## '):
            if current_topic:
                content[current_topic['id']] = '\n'.join(current_content)
            
            topic_match = re.match(r'## (\d+)\.\s*(.*)', line)
            if topic_match:
                topic_id = f"topic-{topic_match.group(1)}"
                topic_title = topic_match.group(2)
                current_topic = {"id": topic_id, "title": topic_title, "number": topic_match.group(1)}
                toc.append(current_topic)
                current_content = [line]
                headlines[topic_id] = []
        
        # Subtopic headlines (#### 1.1 Subtopic)
        elif line.startswith('#### ') and current_topic:
            subtopic_match = re.match(r'#### ([\d.]+)\s*(.*)', line)
            if subtopic_match:
                subtopic_id = f"subtopic-{subtopic_match.group(1).replace('.', '-')}"
                subtopic_title = subtopic_match.group(2)
                headlines[current_topic['id']].append({
                    "id": subtopic_id,
                    "title": subtopic_title
                })
            current_content.append(line)
        
        else:
            if current_topic:
                current_content.append(line)
    
    # Add the last topic
    if current_topic:
        content[current_topic['id']] = '\n'.join(current_content)
    
    return toc, content, headlines

class MCQGenerator:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    def create_mcq_prompt(self, num_questions=5):
        return f"""
        Based on the provided content, generate exactly {num_questions} multiple-choice questions (MCQs).
        
        Requirements:
        1. Create diverse questions covering different aspects of the content
        2. Each question should have 4 options (A, B, C, D)
        3. Only one option should be correct
        4. Make questions challenging but fair
        5. Avoid trivial or overly obvious questions
        
        Format your response as a JSON array with this exact structure:
        [
            {{
                "question": "Question text here?",
                "options": {{
                    "A": "Option A text",
                    "B": "Option B text", 
                    "C": "Option C text",
                    "D": "Option D text"
                }},
                "correct_answer": "A",
                "explanation": "Brief explanation of why this is correct"
            }}
        ]
        
        Generate exactly {num_questions} questions in this format.
        """

    async def upload_to_supabase_storage(self, file_content: bytes, file_name: str):
        try:
            result = supabase.storage.from_(config.SUPABASE_BUCKET).upload(
                file_name, file_content, file_options={"content-type": "auto"}
            )
            return True
        except Exception as e:
            print(f"Error uploading to Supabase storage: {e}")
            return False

    async def generate_mcqs(self, file_content: bytes, file_name: str, num_questions: int = 5):
        try:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            # Upload file to Gemini
            file = genai.upload_file(temp_file_path)
            
            # Wait for file processing
            while file.state.name == "PROCESSING":
                time.sleep(2)
                file = genai.get_file(file.name)
            
            if file.state.name == "FAILED":
                raise ValueError("File processing failed")
            
            # Generate MCQs
            prompt = self.create_mcq_prompt(num_questions)
            response = self.model.generate_content([file, prompt])
            
            # Clean and parse response
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            mcqs = json.loads(response_text)
            
            # Clean up
            genai.delete_file(file.name)
            os.unlink(temp_file_path)
            
            return mcqs
            
        except Exception as e:
            print(f"Error generating MCQs: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating MCQs: {str(e)}")

    async def save_mcqs_to_supabase(self, mcqs: List[Dict], file_name: str, original_filename: str):
        try:
            # Insert quiz record
            quiz_data = {
                "title": f"Quiz from {original_filename}",
                "source_file": file_name,
                "total_questions": len(mcqs),
                "created_at": datetime.now().isoformat()
            }
            
            quiz_result = supabase.table("quizzes").insert(quiz_data).execute()
            quiz_id = quiz_result.data[0]["id"]
            
            # Insert questions
            for i, mcq in enumerate(mcqs):
                question_data = {
                    "quiz_id": quiz_id,
                    "question_number": i + 1,
                    "question_text": mcq["question"],
                    "option_a": mcq["options"]["A"],
                    "option_b": mcq["options"]["B"],
                    "option_c": mcq["options"]["C"],
                    "option_d": mcq["options"]["D"],
                    "correct_answer": mcq["correct_answer"],
                    "explanation": mcq.get("explanation", "")
                }
                
                supabase.table("questions").insert(question_data).execute()
            
            return quiz_id
            
        except Exception as e:
            print(f"Error saving to database: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving to database: {str(e)}")

# Initialize generator
generator = MCQGenerator()

# ================== API Endpoints ==================
# Evaluation API Endpoints
@app.post("/api/evaluate/single", response_model=SingleEvaluationResponse)
async def evaluate_single(
    answer_key: UploadFile = File(...),
    student_response: UploadFile = File(...),
    assignment_id: str = Form(...),
    student_name: str = Form(...)
):
    """Evaluate a single student response."""
    
    try:
        # Read file contents
        logger.info(f"Evaluating single response for student: {student_name}")
        answer_key_content = await answer_key.read()
        student_content = await student_response.read()
        
        # Validate files
        if not answer_key_content or not student_content:
            raise HTTPException(status_code=400, detail="Empty files provided")
        
        # Prepare files for Gemini
        answer_key_mime = get_file_mime_type(answer_key.filename)
        student_mime = get_file_mime_type(student_response.filename)
        
        # Upload files to Gemini
        answer_key_file = genai.upload_file(
            io.BytesIO(answer_key_content),
            mime_type=answer_key_mime,
            display_name=f"answer_key_{answer_key.filename}"
        )
        
        student_file = genai.upload_file(
            io.BytesIO(student_content),
            mime_type=student_mime,
            display_name=f"student_response_{student_response.filename}"
        )
        
        # Wait for file processing
        await asyncio.sleep(2)
        
        # Create evaluation prompt
        prompt = f"""
        You are an expert academic evaluator. Please evaluate the student's response against the provided answer key.

        INSTRUCTIONS:
        1. Compare the student's response with the answer key thoroughly
        2. Identify correct answers, incorrect answers, and areas with partial credit
        3. Provide detailed feedback and suggestions for improvement
        4. Calculate the total marks and percentage score
        5. Assign a letter grade (A, B, C, D, F)
        6. Identify the student's strengths and areas for improvement

        STUDENT: {student_name}

        Return your evaluation in this EXACT JSON format:
        {{
            "total_marks": <integer>,
            "obtained_marks": <integer>,
            "percentage": <float>,
            "grade": "<letter_grade>",
            "correct_answers": ["<answer1>", "<answer2>", ...],
            "incorrect_answers": ["<answer1>", "<answer2>", ...],
            "partial_credit_areas": ["<area1>", "<area2>", ...],
            "strengths": ["<strength1>", "<strength2>", ...],
            "areas_for_improvement": ["<area1>", "<area2>", ...],
            "detailed_feedback": "<comprehensive_feedback_text>"
        }}

        Please analyze both files and provide a thorough evaluation.
        """
        
        # Generate evaluation
        response = eval_model.generate_content([
            prompt,
            answer_key_file,
            student_file
        ])
        
        # Parse response
        response_text = response.text.strip()
        
        # Clean up JSON response (remove markdown formatting if present)
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]
        
        evaluation_data = json.loads(response_text)
        
        # Clean up uploaded files
        genai.delete_file(answer_key_file.name)
        genai.delete_file(student_file.name)
        
        # Save result
        result_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        result = EvaluationResult(
            id=result_id,
            student_name=student_name,
            total_marks=evaluation_data.get('total_marks', 100),
            obtained_marks=evaluation_data.get('obtained_marks', 0),
            percentage=evaluation_data.get('percentage', 0.0),
            grade=evaluation_data.get('grade', 'F'),
            correct_answers=evaluation_data.get('correct_answers', []),
            incorrect_answers=evaluation_data.get('incorrect_answers', []),
            partial_credit_areas=evaluation_data.get('partial_credit_areas', []),
            strengths=evaluation_data.get('strengths', []),
            areas_for_improvement=evaluation_data.get('areas_for_improvement', []),
            detailed_feedback=evaluation_data.get('detailed_feedback', ''),
            timestamp=timestamp,
            evaluation_type="single"
        )
        
        # Save to Supabase
        eval_supabase.table('evaluation_results').insert({
            'id': result_id,
            'assignment_id': assignment_id,
            'student_name': student_name,
            'total_marks': result.total_marks,
            'obtained_marks': result.obtained_marks,
            'percentage': result.percentage,
            'grade': result.grade,
            'correct_answers': result.correct_answers,
            'incorrect_answers': result.incorrect_answers,
            'partial_credit_areas': result.partial_credit_areas,
            'strengths': result.strengths,
            'areas_for_improvement': result.areas_for_improvement,
            'detailed_feedback': result.detailed_feedback,
            'timestamp': timestamp,
            'evaluation_type': "single"
        }).execute()
        
        return SingleEvaluationResponse(
            success=True,
            result=result,
            message="Single evaluation completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single evaluation error: {str(e)}")
        return SingleEvaluationResponse(
            success=False,
            message=f"Evaluation failed: {str(e)}"
        )

@app.post("/api/evaluate/multiple", response_model=MultipleEvaluationResponse)
async def evaluate_multiple(
    answer_key: UploadFile = File(...),
    student_responses: List[UploadFile] = File(...),
    assignment_id: str = Form(...)
):
    """Evaluate multiple student responses."""
    
    try:
        # Read answer key
        answer_key_content = await answer_key.read()
        
        if not answer_key_content:
            raise HTTPException(status_code=400, detail="Empty answer key file")
        
        results = []
        
        # Process each student response
        for student_file in student_responses:
            student_content = await student_file.read()
            
            if not student_content:
                logger.warning(f"Skipping empty file: {student_file.filename}")
                continue
            
            # Extract student name from filename (remove extension)
            student_name = student_file.filename.rsplit('.', 1)[0] if student_file.filename else "Unknown"
            
            try:
                # Prepare files for Gemini
                answer_key_mime = get_file_mime_type(answer_key.filename)
                student_mime = get_file_mime_type(student_file.filename)
                
                # Upload files to Gemini
                answer_key_file = genai.upload_file(
                    io.BytesIO(answer_key_content),
                    mime_type=answer_key_mime,
                    display_name=f"answer_key_{answer_key.filename}"
                )
                
                student_file = genai.upload_file(
                    io.BytesIO(student_content),
                    mime_type=student_mime,
                    display_name=f"student_response_{student_file.filename}"
                )
                
                # Wait for file processing
                await asyncio.sleep(2)
                
                # Create evaluation prompt
                prompt = f"""
                You are an expert academic evaluator. Please evaluate the student's response against the provided answer key.

                INSTRUCTIONS:
                1. Compare the student's response with the answer key thoroughly
                2. Identify correct answers, incorrect answers, and areas with partial credit
                3. Provide detailed feedback and suggestions for improvement
                4. Calculate the total marks and percentage score
                5. Assign a letter grade (A, B, C, D, F)
                6. Identify the student's strengths and areas for improvement

                STUDENT: {student_name}

                Return your evaluation in this EXACT JSON format:
                {{
                    "total_marks": <integer>,
                    "obtained_marks": <integer>,
                    "percentage": <float>,
                    "grade": "<letter_grade>",
                    "correct_answers": ["<answer1>", "<answer2>", ...],
                    "incorrect_answers": ["<answer1>", "<answer2>", ...],
                    "partial_credit_areas": ["<area1>", "<area2>", ...],
                    "strengths": ["<strength1>", "<strength2>", ...],
                    "areas_for_improvement": ["<area1>", "<area2>", ...],
                    "detailed_feedback": "<comprehensive_feedback_text>"
                }}

                Please analyze both files and provide a thorough evaluation.
                """
                
                # Generate evaluation
                response = eval_model.generate_content([
                    prompt,
                    answer_key_file,
                    student_file
                ])
                
                # Parse response
                response_text = response.text.strip()
                
                # Clean up JSON response (remove markdown formatting if present)
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3]
                elif response_text.startswith('```'):
                    response_text = response_text[3:-3]
                
                evaluation_data = json.loads(response_text)
                
                # Clean up uploaded files
                genai.delete_file(answer_key_file.name)
                genai.delete_file(student_file.name)
                
                # Save result
                result_id = str(uuid.uuid4())
                timestamp = datetime.utcnow().isoformat()
                
                result = EvaluationResult(
                    id=result_id,
                    student_name=student_name,
                    total_marks=evaluation_data.get('total_marks', 100),
                    obtained_marks=evaluation_data.get('obtained_marks', 0),
                    percentage=evaluation_data.get('percentage', 0.0),
                    grade=evaluation_data.get('grade', 'F'),
                    correct_answers=evaluation_data.get('correct_answers', []),
                    incorrect_answers=evaluation_data.get('incorrect_answers', []),
                    partial_credit_areas=evaluation_data.get('partial_credit_areas', []),
                    strengths=evaluation_data.get('strengths', []),
                    areas_for_improvement=evaluation_data.get('areas_for_improvement', []),
                    detailed_feedback=evaluation_data.get('detailed_feedback', ''),
                    timestamp=timestamp,
                    evaluation_type="multiple"
                )
                
                # Save to Supabase
                eval_supabase.table('evaluation_results').insert({
                    'id': result_id,
                    'assignment_id': assignment_id,
                    'student_name': student_name,
                    'total_marks': result.total_marks,
                    'obtained_marks': result.obtained_marks,
                    'percentage': result.percentage,
                    'grade': result.grade,
                    'correct_answers': result.correct_answers,
                    'incorrect_answers': result.incorrect_answers,
                    'partial_credit_areas': result.partial_credit_areas,
                    'strengths': result.strengths,
                    'areas_for_improvement': result.areas_for_improvement,
                    'detailed_feedback': result.detailed_feedback,
                    'timestamp': timestamp,
                    'evaluation_type': "multiple"
                }).execute()
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {student_name}: {str(e)}")
                # Continue with other students even if one fails
                continue
        
        if not results:
            raise HTTPException(status_code=400, detail="No valid evaluations completed")
        
        # Calculate summary statistics
        percentages = [r.percentage for r in results]
        grades = [r.grade for r in results]
        
        summary = EvaluationSummary(
            average_percentage=sum(percentages) / len(percentages),
            grade_distribution={grade: grades.count(grade) for grade in set(grades)},
            highest_score=max(percentages),
            lowest_score=min(percentages)
        )
        
        return MultipleEvaluationResponse(
            success=True,
            results=results,
            summary=summary,
            message="Multiple evaluation completed successfully",
            total_students=len(results)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multiple evaluation error: {str(e)}")
        return MultipleEvaluationResponse(
            success=False,
            message=f"Evaluation failed: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/evaluations/assignment/{assignment_id}")
async def get_evaluations(assignment_id: str):
    """Get all evaluations for a specific assignment."""
    try:
        response = eval_supabase.table('evaluation_results').select("*").eq('assignment_id', assignment_id).execute()
        return {"success": True, "data": response.data}
    except Exception as e:
        logger.error(f"Failed to fetch evaluations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch evaluations")

# Tavus Video API Endpoints
@app.post("/api/start-tavus-conversation")
async def start_tavus_conversation(req: TavusConversationRequest):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": TAVUS_API_KEY
    }
    data = {
        "persona_id": "p2a2fc492574",
        "replica_id": "rf4703150052",
        "conversation_name": req.conversation_name,
        "conversational_context": req.conversational_context,
        "custom_greeting": req.custom_greeting,
        "properties": {
            "language": "english"
        }
    }
    try:
        response = requests.post(
            f"{TAVUS_BASE_URL}/conversations",
            headers=headers,
            json=data,
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            conversation_id = result.get("conversation_id")
            conversation_url = result.get("conversation_url")
            if not conversation_id or not conversation_url:
                logger.error("Tavus API did not return conversation_id or conversation_url. Response:", result)
                raise HTTPException(
                    status_code=500,
                    detail=f"Tavus API did not return conversation_id or conversation_url. Response: {result}"
                )
            return {"conversation_id": conversation_id, "conversation_url": conversation_url, "status": "initialized"}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Tavus API error: {response.text}"
            )
    except Exception as e:
        logger.error(f"Tavus conversation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tavus conversation error: {str(e)}")

@app.post("/api/end-tavus-conversation")
async def end_tavus_conversation(req: EndConversationRequest):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": TAVUS_API_KEY
    }
    try:
        response = requests.delete(
            f"{TAVUS_BASE_URL}/conversations/{req.conversation_id}",
            headers=headers,
            timeout=30
        )
        if response.status_code == 200:
            return {"status": "conversation_ended", "conversation_id": req.conversation_id}
        else:
            logger.error(f"Failed to end conversation {req.conversation_id}. Status: {response.status_code}, Response: {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to end Tavus conversation: {response.text}"
            )
    except Exception as e:
        logger.error(f"Error ending Tavus conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ending conversation: {str(e)}")

# Course Generator API Endpoints
@app.post("/generate-course", response_model=CourseResponse)
async def generate_course(request: CourseRequest):
    try:
        logger.info(f"Received request for subject: {request.subject}")
        
        prompt = f"""
        You are a master curriculum designer. Generate a **complete {request.subject} course** in **Markdown** format.
        
        Structure:
        1. **Table of Contents**: Numbered list of 8-12 main topics
        2. **Main Content**: For each topic:
           - Second-level heading: ## 1. Topic Name
           - ### Description (150-200 words)
           - ### Subtopics (bullet list)
           - #### 1.1 Subtopic Name (1000-1500 words + code examples)
           - #### 1.2 Next Subtopic (continue pattern)
           - Exercises after each subtopic
        
        Cover essential {request.subject} topics with proper progression.
        Use consistent Markdown formatting with proper heading hierarchy.
        Include code examples in fenced blocks where applicable.
        """
        
        logger.info("Generating course content...")
        markdown_content = generate_markdown(prompt)
        logger.info(f"Generated content length: {len(markdown_content)}")
        
        logger.info("Parsing course content...")
        toc, content, headlines = parse_course_content(markdown_content)
        logger.info(f"Parsed TOC items: {len(toc)}")
        
        return CourseResponse(
            toc=toc,
            content=content,
            headlines=headlines
        )
    
    except Exception as e:
        logger.error(f"Error generating course: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# MCQ Generator API Endpoints
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    num_questions: int = Form(5)
):
    try:
        # Validate file type
        allowed_types = ["application/pdf", "text/plain", "image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Validate number of questions
        if not 1 <= num_questions <= 20:
            raise HTTPException(status_code=400, detail="Number of questions must be between 1 and 20")
        
        # Read file content
        file_content = await file.read()
        
        # Generate unique filename for storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = os.path.splitext(file.filename)[1]
        storage_filename = f"{timestamp}_{file.filename}"
        
        # Upload to Supabase storage
        await generator.upload_to_supabase_storage(file_content, storage_filename)
        
        # Generate MCQs
        mcqs = await generator.generate_mcqs(file_content, file.filename, num_questions)
        
        # Save to database
        quiz_id = await generator.save_mcqs_to_supabase(mcqs, storage_filename, file.filename)
        
        return {
            "message": "File uploaded and MCQs generated successfully",
            "quiz_id": quiz_id,
            "total_questions": len(mcqs)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/quiz/{quiz_id}")
async def get_quiz(quiz_id: int):
    try:
        # Get quiz info
        quiz_result = supabase.table("quizzes").select("*").eq("id", quiz_id).execute()
        if not quiz_result.data:
            raise HTTPException(status_code=404, detail="Quiz not found")
        
        quiz = quiz_result.data[0]
        
        # Get questions
        questions_result = supabase.table("questions").select("*").eq("quiz_id", quiz_id).order("question_number").execute()
        
        # Format questions for frontend
        questions = []
        for q in questions_result.data:
            questions.append({
                "id": q["id"],
                "question_number": q["question_number"],
                "question_text": q["question_text"],
                "options": {
                    "A": q["option_a"],
                    "B": q["option_b"],
                    "C": q["option_c"],
                    "D": q["option_d"]
                },
                "correct_answer": q["correct_answer"],
                "explanation": q["explanation"]
            })
        
        return {
            "quiz": quiz,
            "questions": questions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz/{quiz_id}/submit")
async def submit_quiz(quiz_id: int, answers: Dict[str, str]):
    try:
        # Get correct answers
        questions_result = supabase.table("questions").select("*").eq("quiz_id", quiz_id).execute()
        
        correct_answers = {}
        explanations = {}
        for q in questions_result.data:
            correct_answers[str(q["id"])] = q["correct_answer"]
            explanations[str(q["id"])] = q["explanation"]
        
        # Calculate score
        score = 0
        results = {}
        
        for question_id, user_answer in answers.items():
            is_correct = user_answer == correct_answers.get(question_id)
            if is_correct:
                score += 1
            
            results[question_id] = {
                "user_answer": user_answer,
                "correct_answer": correct_answers.get(question_id),
                "is_correct": is_correct,
                "explanation": explanations.get(question_id)
            }
        
        return {
            "score": score,
            "total_questions": len(correct_answers),
            "percentage": round((score / len(correct_answers)) * 100, 2),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quizzes")
async def get_recent_quizzes():
    try:
        result = supabase.table("quizzes").select("*").order("created_at", desc=True).limit(10).execute()
        return {"quizzes": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Combined EduTech API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
