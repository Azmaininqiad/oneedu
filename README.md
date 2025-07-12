# oneedu Backend Documentation

## Overview

This is the backend codebase for the [OneEdu Classroom platform](https://classroom-3ymf.vercel.app/). It provides APIs for:
- Automated evaluation of student assignments
- MCQ (Multiple Choice Question) generation from uploaded files
- Course content generation
- Video-based AI conversations (Tavus integration)
- Quiz management and scoring

The backend is built with **FastAPI** and integrates with **Supabase** (database & storage), **Google Gemini** (AI/LLM), **OpenRouter** (AI/LLM), and **Tavus** (video AI).

---

## Tech Stack

- **Language:** Python 3
- **Framework:** FastAPI
- **Database & Storage:** Supabase
- **AI/LLM:** Google Gemini, OpenRouter
- **Video AI:** Tavus
- **Other:** Uvicorn (ASGI server), Pydantic, Requests, python-dotenv

### Key Dependencies (see `requirements.txt`)
- fastapi
- uvicorn
- pydantic
- requests
- google-generativeai
- supabase
- python-multipart
- python-jose
- passlib[bcrypt]
- python-decouple
- python-dotenv

---

## Workflow & API Endpoints

### 1. Evaluation APIs
- **/api/evaluate/single**: Evaluate a single student's response against an answer key (file upload). Returns detailed feedback, marks, grade, strengths, and areas for improvement. Stores results in Supabase.
- **/api/evaluate/multiple**: Batch evaluation for multiple students. Returns summary statistics and stores all results.
- **/evaluations/assignment/{assignment_id}**: Fetch all evaluations for a given assignment.

### 2. MCQ Generator APIs
- **/upload**: Upload a file (PDF, TXT, image) and generate MCQs using Gemini. Stores quiz and questions in Supabase.
- **/quiz/{quiz_id}**: Retrieve quiz and questions by ID.
- **/quiz/{quiz_id}/submit**: Submit answers for a quiz and get score, correct answers, and explanations.
- **/quizzes**: Get recent quizzes.

### 3. Course Generator API
- **/generate-course**: Generate a full course (table of contents, topics, subtopics, exercises) for a given subject using OpenRouter LLM.

### 4. Tavus Video Conversation APIs
- **/api/start-tavus-conversation**: Start a video-based AI conversation (Tavus integration).
- **/api/end-tavus-conversation**: End a Tavus conversation.

### 5. Miscellaneous
- **/api/health**: Health check endpoint.
- **/**: Root endpoint (API status message).

---

## Code Structure

- `main.py`: Main FastAPI app, all endpoints, models, and integrations.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: (if present) Containerization for deployment.

### Key Classes & Functions
- **Config**: Stores API keys and config values.
- **Pydantic Models**: For request/response validation (EvaluationResult, CourseRequest, etc).
- **MCQGenerator**: Handles MCQ prompt creation, file upload, MCQ generation, and database storage.
- **Helper Functions**: For file type detection, grade calculation, markdown parsing, etc.

---

## Integrations

- **Supabase**: Used for storing evaluation results, quizzes, and questions. Also used for file storage (PDFs, images, etc).
- **Google Gemini**: Used for AI-powered evaluation and MCQ generation from files.
- **OpenRouter**: Used for generating course content in markdown.
- **Tavus**: Used for video-based AI conversations.

---

## Environment Variables

Sensitive API keys and URLs are currently hardcoded in `main.py` (should be moved to environment variables for production).

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
3. The API will be available at `http://localhost:8000/`

---

## Notes
- This backend is designed to work with the [OneEdu Classroom frontend](https://classroom-3ymf.vercel.app/).
- For production, move all API keys and secrets to environment variables and use `python-dotenv` or similar.
- See the code in `main.py` for detailed logic and endpoint implementations.
