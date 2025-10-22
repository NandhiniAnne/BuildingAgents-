from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import uuid
from datetime import datetime

# CrewAI and AI imports
from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.embeddings import OllamaEmbeddings

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Document processing
import PyPDF2
import docx
import re
import json

# Configuration
OLLAMA_MODEL = "mistral:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "resumes_collection"
UPLOAD_DIR = "uploaded_resumes"

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Resume Intelligence API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

# Initialize LLM for agents with optimized settings for Mistral
llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    timeout=900,  # 15 minutes timeout
    temperature=0.1,  # Lower temperature for faster, more focused responses
    max_tokens=500  # Limit response length
)

# Create collection if not exists
try:
    # Delete existing collection to fix dimension issues
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass
    
    test_emb = embeddings.embed_query("test")
    VECTOR_SIZE = len(test_emb)
    print(f"📏 Detected vector size: {VECTOR_SIZE}")
    
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"✅ Created Qdrant collection: {COLLECTION_NAME}")
except Exception as e:
    print(f"❌ Collection error: {e}")


# Pydantic models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class ResumeData(BaseModel):
    id: str
    filename: str
    name: str
    email: str
    phone: str
    skills: List[str]
    experience: str
    uploaded_at: str


# ============== CREWAI AGENTS ==============

# Agent 1: Resume Parser Agent
resume_parser_agent = Agent(
    role="Resume Data Extraction Specialist",
    goal="Extract structured information from resume text including name, email, phone, and experience",
    backstory="""You are an expert at parsing resumes and extracting key information accurately. 
    You understand various resume formats and can identify contact details and experience.
    Return information in a structured format.""",
    llm=llm,
    verbose=True  # ← Changed to True to see output
)

# Agent 2: Skills Analyzer Agent (Optimized for Mistral)
skills_analyzer_agent = Agent(
    role="Technical Skills Analyst",
    goal="Extract technical skills as a comma-separated list",
    backstory="""You extract technical skills from resumes. 
    Return ONLY a comma-separated list. Be concise and fast.""",
    llm=llm,
    verbose=True,
    max_iter=3,  # Limit iterations to prevent endless loops
    allow_delegation=False  # Don't delegate to other agents
)

# Agent 3: Query Assistant Agent (Optimized for Mistral)
query_assistant_agent = Agent(
    role="Resume Query Assistant",
    goal="Answer questions about resumes in 2-3 sentences",
    backstory="""You answer questions about candidate resumes clearly and concisely. 
    Keep responses brief and factual.""",
    llm=llm,
    verbose=True,
    max_iter=3,
    allow_delegation=False
)


# ============== UTILITY FUNCTIONS ==============

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Error: {str(e)}"


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error: {str(e)}"


def extract_resume_info_with_agents(text: str, filename: str) -> dict:
    """Use CrewAI agents to extract resume information - OPTIMIZED VERSION"""
    
    print(f"\n{'='*60}")
    print(f"🤖 PROCESSING RESUME: {filename}")
    print(f"{'='*60}\n")
    
    # ============== QUICK EXTRACTION WITH REGEX ==============
    print("⚡ Quick extraction with regex for basic info...")
    
    # Email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    email = emails[0] if emails else "Not found"
    
    # Phone
    phone_patterns = [
        r'(?:Mob|Mobile|Phone|Tel|Cell)[\s#:]*([+]?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    ]
    phone = "Not found"
    for pattern in phone_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            phone = matches[0] if isinstance(matches[0], str) else ''.join(matches[0])
            phone = re.sub(r'[^\d+\-\s()]', '', phone).strip()
            break
    
    # Name
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    name = "Not found"
    for line in lines[:10]:
        if any(skip in line.lower() for skip in ['objective', 'email', 'phone', 'address', 'po box', 'summary']):
            continue
        words = line.split()
        if 2 <= len(words) <= 5 and all(word[0].isupper() for word in words[:3] if word):
            name = re.sub(r',.*', '', line)
            name = re.sub(r'\s+(MBA|MS|PMP|CISSP|PhD).*', '', name, flags=re.IGNORECASE)
            break
    
    if name == "Not found" and email != "Not found":
        name = email.split('@')[0].replace('.', ' ').replace('_', ' ').title()
    
    # Experience
    exp_patterns = [
        r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+)\+?\s*years?'
    ]
    experience = "Not specified"
    for pattern in exp_patterns:
        match = re.search(pattern, text.lower())
        if match:
            experience = f"{match.group(1)} years"
            break
    
    print(f"✅ Basic info extracted: {name} | {email}")
    
    # ============== AGENT: Skills Analyzer (Only This One) ==============
    print("\n💡 AGENT: Technical Skills Analyst - Extracting skills...")
    
    skills_task = Task(
        description=f"""List technical skills from this resume as comma-separated values.

Resume:
{text[:800]}

Extract: certifications, programming languages, tools, frameworks.
Return format: skill1, skill2, skill3
Maximum 10 skills.""",
        expected_output="Comma-separated technical skills list",
        agent=skills_analyzer_agent
    )
    
    skills_crew = Crew(
        agents=[skills_analyzer_agent],
        tasks=[skills_task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        print("🚀 Running Skills Analyzer Agent...\n")
        result = skills_crew.kickoff()
        skills_text = str(result)
        skills = [s.strip() for s in skills_text.split(',') if s.strip()][:15]
        print(f"\n✅ Skills extracted: {len(skills)} skills found")
    except Exception as e:
        print(f"❌ Agent error: {e}")
        skills = []
    
    print(f"\n{'='*60}")
    print(f"✅ RESUME PROCESSING COMPLETE")
    print(f"{'='*60}\n")
    
    return {
        "filename": filename,
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "experience": experience,
        "full_text": text[:2000]
    }


# ============== API ENDPOINTS ==============

@app.get("/")
async def root():
    return {"message": "Resume Intelligence API", "status": "running"}


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        count = collection_info.points_count
        return {
            "total_resumes": count,
            "collection": COLLECTION_NAME,
            "status": "connected"
        }
    except Exception as e:
        return {"total_resumes": 0, "status": "error", "message": str(e)}


@app.post("/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    """Upload and process resume files with ALL AGENTS"""
    
    processed_resumes = []
    points = []
    
    for file in files:
        # Save file
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Extract text
        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            text = extract_text_from_docx(file_path)
        else:
            continue
        
        if "Error" in text:
            continue
        
        # Extract info using MULTIPLE agents
        resume_info = extract_resume_info_with_agents(text, file.filename)
        resume_info['id'] = file_id
        resume_info['uploaded_at'] = datetime.now().isoformat()
        
        # Create embedding
        embedding_text = f"Name: {resume_info['name']}\nEmail: {resume_info['email']}\nSkills: {', '.join(resume_info['skills'])}"
        embedding = embeddings.embed_query(embedding_text)
        
        # Store in Qdrant
        point = PointStruct(
            id=file_id,
            vector=embedding,
            payload=resume_info
        )
        points.append(point)
        processed_resumes.append(resume_info)
    
    # Upload to Qdrant
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
    
    return {
        "message": f"Successfully processed {len(processed_resumes)} resumes",
        "resumes": processed_resumes
    }


@app.get("/resumes")
async def get_all_resumes():
    """Get all stored resumes"""
    try:
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100
        )
        
        resumes = []
        for point in results[0]:
            resume_data = point.payload
            resume_data['id'] = str(point.id)
            resumes.append(resume_data)
        
        return {"resumes": resumes, "count": len(resumes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_resumes(request: QueryRequest):
    """Query resumes using Query Assistant Agent"""
    
    print(f"\n{'='*60}")
    print(f"🔍 NEW QUERY: {request.query}")
    print(f"{'='*60}\n")
    
    # Create query embedding
    query_embedding = embeddings.embed_query(request.query)
    
    # Search in Qdrant using the correct method
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=5
    )
    
    if not search_results:
        return QueryResponse(
            answer="No matching resumes found in the database.",
            sources=[]
        )
    
    # Prepare context for Query Assistant Agent
    context = "Relevant resumes found:\n\n"
    sources = []
    
    for i, result in enumerate(search_results[:3], 1):
        resume = result.payload
        context += f"{i}. {resume.get('name', 'N/A')}\n"
        context += f"   Email: {resume.get('email', 'N/A')}\n"
        context += f"   Phone: {resume.get('phone', 'N/A')}\n"
        context += f"   Skills: {', '.join(resume.get('skills', []))}\n"
        context += f"   Experience: {resume.get('experience', 'Not specified')}\n\n"
        
        sources.append({
            "name": resume.get('name', 'N/A'),
            "email": resume.get('email', 'N/A'),
            "phone": resume.get('phone', 'N/A'),
            "skills": resume.get('skills', []),
            "score": result.score
        })
    
    print("💬 AGENT 3: Query Assistant Agent - Answering query...\n")
    
    # Create task for Query Assistant Agent (Shorter prompt for Mistral)
    query_task = Task(
        description=f"""Answer: "{request.query}"

Data:
{context}

Provide a brief 2-3 sentence answer with specific names and details.""",
        expected_output="Brief answer with candidate details",
        agent=query_assistant_agent
    )
    
    # Execute with CrewAI
    query_crew = Crew(
        agents=[query_assistant_agent],
        tasks=[query_task],
        process=Process.sequential,
        verbose=True  # ← Changed to True to see agent output
    )
    
    result = query_crew.kickoff()
    answer = str(result)
    
    print(f"\n✅ Query Assistant Answer:\n{answer}\n")
    print(f"{'='*60}\n")
    
    return QueryResponse(answer=answer, sources=sources)


@app.delete("/resumes/clear")
async def clear_all_resumes():
    """Clear all resumes from database"""
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
        
        # Recreate collection
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        
        return {"message": "All resumes cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting Resume Intelligence API Server...")
    print(f"📊 Qdrant: {QDRANT_URL}")
    print(f"🤖 Ollama Model: {OLLAMA_MODEL}")
    print(f"🔢 Embedding Model: {EMBEDDING_MODEL}")
    print(f"📁 Upload Directory: {UPLOAD_DIR}")
    print(f"📏 Vector Size: {VECTOR_SIZE}")
    print("\n" + "="*60)
    print("🎯 ALL 3 AGENTS ARE NOW ACTIVE AND VERBOSE!")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)