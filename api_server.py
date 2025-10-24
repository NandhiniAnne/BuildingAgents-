from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import uuid
from datetime import datetime
import json

# CrewAI and AI imports
from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.embeddings import OllamaEmbeddings
from crewai.tools import tool

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Document processing
import PyPDF2
import docx
import re

# Configuration
OLLAMA_MODEL = "mistral:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "resumes_collection"
UPLOAD_DIR = "uploaded_resumes"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Master AI Resume Intelligence Chatbot")

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

# Initialize LLM
llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    timeout=900,
    temperature=0.1,
    max_tokens=2000  # Larger for comprehensive extraction
)

# Create collection
try:
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
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    agent_used: str
    sources: Optional[List[dict]] = None


# ============== SINGLE MASTER EXTRACTION AGENT ==============

master_extraction_agent = Agent(
    role="Master Resume Data Extraction Specialist",
    goal="Extract ALL information from resumes in structured JSON format",
    backstory="""You are an expert resume parser that extracts comprehensive information.

You extract:
1. CONTACT INFO: Name, Email, Phone, Location (City, State), LinkedIn URL, Portfolio/GitHub URL
2. PROFESSIONAL: Current Role/Title, Years of Experience (numeric), Previous Companies (list), Salary Expectations
3. TECHNICAL SKILLS: Programming languages, frameworks, tools, cloud platforms, databases
4. CERTIFICATIONS: AWS Certified, PMP, CISSP, Azure, Google Cloud, etc.
5. EDUCATION: Degree, Major, University, Graduation Year

CRITICAL RULES:
- Extract data EXACTLY as it appears
- For Name: Remove ALL certifications (PMP, MBA, PhD, etc.)
- For Location: Format as "City, State" (e.g., "Phoenix, Arizona")
- For Skills: List technical skills only (max 15)
- For Years of Experience: Extract numeric value (e.g., 5, 10, 3-5)
- For Previous Companies: List up to 5 companies
- Return ONLY valid JSON format

HOW SKILLS ARE EXTRACTED:
1. Look in "Skills" or "Technical Skills" sections
2. Identify programming languages: Python, Java, JavaScript, C++, Go, Ruby, PHP
3. Identify frameworks: React, Angular, Django, Spring, Node.js, Flask
4. Identify cloud: AWS, Azure, GCP, Cloud Computing
5. Identify databases: MySQL, PostgreSQL, MongoDB, Redis, Oracle
6. Identify tools: Docker, Kubernetes, Git, Jenkins, Terraform, CI/CD
7. Identify other tech: Machine Learning, AI, Data Science, APIs, Microservices

EXTRACTION PROCESS:
- Read the entire resume carefully
- Look for contact info at the top
- Find skills in dedicated skills section or throughout experience
- Extract certifications from certifications section or after name
- Get education from education section (usually at bottom)
- Count years from work history dates or "X years of experience" statements""",
    llm=llm,
    verbose=True,
    max_iter=3,
    allow_delegation=False
)


# ============== EXTRACTION FUNCTION USING SINGLE AGENT ==============

def extract_resume_info_with_single_agent(text: str, filename: str) -> dict:
    """
    Use SINGLE master agent to extract ALL information
    Returns structured data ready for database storage
    """
    
    print(f"\n{'='*60}")
    print(f"🤖 PROCESSING RESUME: {filename}")
    print(f"{'='*60}\n")
    
    # Prepare resume text (take more content for better extraction)
    resume_text = text[:3000]
    
    # Create comprehensive extraction task
    extraction_task = Task(
        description=f"""Extract ALL information from this resume and return as JSON.

RESUME TEXT:
{resume_text}

EXTRACT THE FOLLOWING (return valid JSON):

{{
  "name": "First Last (remove PMP, MBA, etc.)",
  "email": "email@example.com",
  "phone": "123-456-7890",
  "location": "City, State",
  "linkedin": "URL or Not found",
  "portfolio": "URL or Not found",
  "current_role": "Job Title",
  "years_experience": 5,
  "previous_companies": ["Company1", "Company2", "Company3"],
  "salary_expectations": "$120K or Not found",
  "skills": ["Python", "AWS", "Docker", "React", "etc"],
  "certifications": ["AWS Certified", "PMP", "etc"],
  "degree": "BS in Computer Science",
  "university": "University Name",
  "graduation_year": "2020 or Not found"
}}

IMPORTANT:
- Return ONLY the JSON object, no extra text
- Use "Not found" for missing fields
- years_experience must be a NUMBER (0 if not found)
- Skills: technical skills only, max 15
- Certifications: empty array [] if none found
- Extract location as "City, State" format
- Remove certifications from name field

SKILLS EXTRACTION GUIDE:
Look for these patterns:
- Programming: Python, Java, JavaScript, C++, C#, Go, Ruby, PHP, Swift, Kotlin
- Web: React, Angular, Vue, Node.js, Django, Flask, Spring, ASP.NET
- Cloud: AWS, Azure, GCP, Cloud Computing, Serverless
- Databases: MySQL, PostgreSQL, MongoDB, Redis, Oracle, SQL Server
- DevOps: Docker, Kubernetes, Jenkins, CI/CD, Terraform, Ansible
- Data: Machine Learning, AI, Data Science, Pandas, TensorFlow, PyTorch
- Other: Git, REST API, Microservices, Agile, Linux

Return the JSON now:""",
        expected_output="Valid JSON object with all extracted resume fields",
        agent=master_extraction_agent
    )
    
    # Execute extraction
    crew = Crew(
        agents=[master_extraction_agent],
        tasks=[extraction_task],
        process=Process.sequential,
        verbose=False
    )
    
    try:
        print("🔄 Master agent extracting data...")
        result = crew.kickoff()
        
        # Get the raw output
        if hasattr(result, 'tasks_output'):
            raw_output = str(result.tasks_output[0].raw)
        else:
            raw_output = str(result)
        
        print(f"📄 Raw output:\n{raw_output[:500]}...\n")
        
        # Parse JSON from output
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            json_str = json_match.group(0)
            extracted_data = json.loads(json_str)
        else:
            # Fallback: try parsing entire response
            extracted_data = json.loads(raw_output)
        
        # Validate and set defaults
        resume_info = {
            "filename": filename,
            "name": extracted_data.get("name", "Not found"),
            "email": extracted_data.get("email", "Not found"),
            "phone": extracted_data.get("phone", "Not found"),
            "location": extracted_data.get("location", "Not found"),
            "linkedin": extracted_data.get("linkedin", "Not found"),
            "portfolio": extracted_data.get("portfolio", "Not found"),
            "current_role": extracted_data.get("current_role", "Not specified"),
            "years_experience": int(extracted_data.get("years_experience", 0)) if isinstance(extracted_data.get("years_experience"), (int, float, str)) else 0,
            "previous_companies": extracted_data.get("previous_companies", [])[:5],
            "salary_expectations": extracted_data.get("salary_expectations", "Not specified"),
            "skills": extracted_data.get("skills", [])[:15],
            "certifications": extracted_data.get("certifications", [])[:10],
            "degree": extracted_data.get("degree", "Not found"),
            "university": extracted_data.get("university", "Not found"),
            "graduation_year": extracted_data.get("graduation_year", "Not found"),
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Clean years_experience if it's a string
        if isinstance(resume_info['years_experience'], str):
            numbers = re.findall(r'\d+', resume_info['years_experience'])
            resume_info['years_experience'] = int(numbers[0]) if numbers else 0
        
        print(f"✅ EXTRACTED DATA:")
        print(f"   👤 Name: {resume_info['name']}")
        print(f"   📧 Email: {resume_info['email']}")
        print(f"   📍 Location: {resume_info['location']}")
        print(f"   💼 Role: {resume_info['current_role']}")
        print(f"   📅 Experience: {resume_info['years_experience']} years")
        print(f"   🛠️  Skills ({len(resume_info['skills'])}): {', '.join(resume_info['skills'][:5])}...")
        print(f"   🏆 Certifications ({len(resume_info['certifications'])}): {', '.join(resume_info['certifications'][:3])}...")
        print(f"   🎓 Education: {resume_info['degree']}")
        print(f"   🏢 Companies: {', '.join(resume_info['previous_companies'][:3])}")
        
        return resume_info
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Raw output was: {raw_output}")
        return create_default_resume_info(filename)
    
    except Exception as e:
        print(f"❌ Extraction Error: {e}")
        return create_default_resume_info(filename)


def create_default_resume_info(filename: str) -> dict:
    """Create default structure when extraction fails"""
    return {
        "filename": filename,
        "name": "Not found",
        "email": "Not found",
        "phone": "Not found",
        "location": "Not found",
        "linkedin": "Not found",
        "portfolio": "Not found",
        "current_role": "Not specified",
        "years_experience": 0,
        "previous_companies": [],
        "salary_expectations": "Not specified",
        "skills": [],
        "certifications": [],
        "degree": "Not found",
        "university": "Not found",
        "graduation_year": "Not found",
        "uploaded_at": datetime.now().isoformat()
    }


def create_rich_embedding_text(resume_info: dict) -> str:
    """Create comprehensive embedding text for better semantic search"""
    
    parts = [
        f"Name: {resume_info['name']}",
        f"Location: {resume_info['location']}",
        f"Current Role: {resume_info['current_role']}",
        f"Experience: {resume_info['years_experience']} years",
    ]
    
    if resume_info['previous_companies']:
        parts.append(f"Worked at: {', '.join(resume_info['previous_companies'])}")
    
    if resume_info['skills']:
        parts.append(f"Skills: {', '.join(resume_info['skills'])}")
    
    if resume_info['certifications']:
        parts.append(f"Certifications: {', '.join(resume_info['certifications'])}")
    
    parts.append(f"Education: {resume_info['degree']} from {resume_info['university']}")
    
    return "\n".join(parts)


# ============== TOOLS FOR SEARCH AGENT ==============

@tool("Advanced Search Resumes")
def advanced_search_tool(query: str) -> str:
    """
    Advanced semantic search with filters
    Args:
        query: Natural language query (e.g., "Python developers in Phoenix with AWS")
    Returns:
        Formatted list of matching candidates
    """
    try:
        # Extract filters from query using simple patterns
        location_filter = None
        min_years = None
        
        # Location patterns
        location_keywords = ["in ", "from ", "located in ", "based in "]
        for keyword in location_keywords:
            if keyword in query.lower():
                parts = query.lower().split(keyword)
                if len(parts) > 1:
                    location_part = parts[1].split()[0:3]  # Take next few words
                    location_filter = ' '.join(location_part).strip('.,!?')
        
        # Experience patterns
        exp_patterns = [r'(\d+)\+?\s*years?', r'minimum\s+(\d+)\s+years?', r'at least\s+(\d+)\s+years?']
        for pattern in exp_patterns:
            match = re.search(pattern, query.lower())
            if match:
                min_years = int(match.group(1))
                break
        
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=20
        )
        
        if not search_results:
            return "No matching candidates found."
        
        # Apply filters
        filtered_results = []
        for result in search_results:
            resume = result.payload
            
            # Location filter
            if location_filter:
                if location_filter.lower() not in resume.get('location', '').lower():
                    continue
            
            # Experience filter
            if min_years:
                if resume.get('years_experience', 0) < min_years:
                    continue
            
            filtered_results.append(result)
        
        if not filtered_results:
            return f"No candidates found matching: {query}\nTry broadening your search criteria."
        
        # Format results
        results_text = f"🎯 Found {len(filtered_results)} Matching Candidates:\n\n"
        
        for i, result in enumerate(filtered_results[:5], 1):
            resume = result.payload
            results_text += f"{'='*50}\n"
            results_text += f"#{i} - {resume.get('name', 'N/A')}\n"
            results_text += f"{'='*50}\n"
            results_text += f"📧 Email: {resume.get('email', 'N/A')}\n"
            results_text += f"📱 Phone: {resume.get('phone', 'N/A')}\n"
            results_text += f"📍 Location: {resume.get('location', 'N/A')}\n"
            results_text += f"💼 Current Role: {resume.get('current_role', 'N/A')}\n"
            results_text += f"📅 Experience: {resume.get('years_experience', 0)} years\n"
            
            skills = resume.get('skills', [])
            if skills:
                results_text += f"🛠️  Skills: {', '.join(skills[:8])}\n"
            
            certs = resume.get('certifications', [])
            if certs:
                results_text += f"🏆 Certifications: {', '.join(certs)}\n"
            
            results_text += f"🎓 Education: {resume.get('degree', 'N/A')}\n"
            
            companies = resume.get('previous_companies', [])
            if companies:
                results_text += f"🏢 Previous: {', '.join(companies[:3])}\n"
            
            if resume.get('linkedin', 'Not found') != 'Not found':
                results_text += f"💼 LinkedIn: {resume.get('linkedin')}\n"
            
            results_text += f"✓ Match Score: {result.score*100:.1f}%\n\n"
        
        if len(filtered_results) > 5:
            results_text += f"... and {len(filtered_results) - 5} more candidates\n"
        
        return results_text
    
    except Exception as e:
        return f"Error searching: {str(e)}"


@tool("Get Resume Count")
def get_resume_count_tool() -> str:
    """Get total number of resumes in database"""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        count = collection_info.points_count
        return f"📊 Total resumes in database: {count}"
    except Exception as e:
        return f"Error getting count: {str(e)}"


@tool("List All Resumes")
def list_all_resumes_tool() -> str:
    """List all candidates in database"""
    try:
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=20
        )
        
        if not results[0]:
            return "No resumes in database."
        
        resumes_text = "📋 All Candidates:\n\n"
        for i, point in enumerate(results[0], 1):
            resume = point.payload
            resumes_text += f"{i}. {resume.get('name', 'N/A')} - {resume.get('current_role', 'N/A')}\n"
            resumes_text += f"   📍 {resume.get('location', 'N/A')} | 📅 {resume.get('years_experience', 0)} years\n"
            resumes_text += f"   🛠️  {', '.join(resume.get('skills', [])[:4])}\n\n"
        
        return resumes_text
    
    except Exception as e:
        return f"Error listing resumes: {str(e)}"


@tool("Get Candidate Details")
def get_candidate_details_tool(name: str) -> str:
    """Get detailed info about a specific candidate"""
    try:
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100
        )
        
        for point in results[0]:
            resume = point.payload
            if name.lower() in resume.get('name', '').lower():
                details = f"👤 {resume.get('name', 'N/A')}\n"
                details += f"{'='*50}\n"
                details += f"📧 Email: {resume.get('email', 'N/A')}\n"
                details += f"📱 Phone: {resume.get('phone', 'N/A')}\n"
                details += f"📍 Location: {resume.get('location', 'N/A')}\n"
                details += f"💼 Current Role: {resume.get('current_role', 'N/A')}\n"
                details += f"📅 Experience: {resume.get('years_experience', 0)} years\n"
                details += f"🛠️  Skills: {', '.join(resume.get('skills', []))}\n"
                
                certs = resume.get('certifications', [])
                if certs:
                    details += f"🏆 Certifications: {', '.join(certs)}\n"
                
                details += f"🎓 Education: {resume.get('degree', 'N/A')} - {resume.get('university', 'N/A')}\n"
                
                companies = resume.get('previous_companies', [])
                if companies:
                    details += f"🏢 Previous Companies: {', '.join(companies)}\n"
                
                if resume.get('linkedin', 'Not found') != 'Not found':
                    details += f"💼 LinkedIn: {resume.get('linkedin')}\n"
                if resume.get('portfolio', 'Not found') != 'Not found':
                    details += f"🌐 Portfolio: {resume.get('portfolio')}\n"
                
                return details
        
        return f"Candidate '{name}' not found."
    
    except Exception as e:
        return f"Error: {str(e)}"


@tool("Compare Two Candidates")
def compare_candidates_tool(name1: str, name2: str) -> str:
    """Compare two candidates side-by-side"""
    try:
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100
        )
        
        candidate1 = None
        candidate2 = None
        
        for point in results[0]:
            resume = point.payload
            name = resume.get('name', '').lower()
            if name1.lower() in name and not candidate1:
                candidate1 = resume
            if name2.lower() in name and not candidate2:
                candidate2 = resume
            if candidate1 and candidate2:
                break
        
        if not candidate1 or not candidate2:
            missing = []
            if not candidate1: missing.append(name1)
            if not candidate2: missing.append(name2)
            return f"Could not find: {', '.join(missing)}"
        
        comparison = f"""
{'='*60}
CANDIDATE COMPARISON
{'='*60}

{candidate1['name']:30} vs {candidate2['name']}
{'-'*60}
📧 Email:        {candidate1.get('email', 'N/A'):30} | {candidate2.get('email', 'N/A')}
📍 Location:     {candidate1.get('location', 'N/A'):30} | {candidate2.get('location', 'N/A')}
💼 Role:         {candidate1.get('current_role', 'N/A'):30} | {candidate2.get('current_role', 'N/A')}
📅 Experience:   {candidate1.get('years_experience', 0)} years{' '*(24)} | {candidate2.get('years_experience', 0)} years
🎓 Education:    {candidate1.get('degree', 'N/A'):30} | {candidate2.get('degree', 'N/A')}

🛠️  SKILLS:
Candidate 1: {', '.join(candidate1.get('skills', [])[:10])}
Candidate 2: {', '.join(candidate2.get('skills', [])[:10])}

Common Skills: {', '.join(set(candidate1.get('skills', [])) & set(candidate2.get('skills', [])))}

🏆 CERTIFICATIONS:
Candidate 1: {', '.join(candidate1.get('certifications', [])) if candidate1.get('certifications') else 'None'}
Candidate 2: {', '.join(candidate2.get('certifications', [])) if candidate2.get('certifications') else 'None'}
"""
        return comparison
    
    except Exception as e:
        return f"Error comparing: {str(e)}"


# ============== SINGLE MASTER CONVERSATIONAL AGENT ==============

master_chat_agent = Agent(
    role="Master Resume Intelligence Assistant",
    goal="Help users with ALL resume tasks through natural conversation",
    backstory="""You are an intelligent AI assistant for resume management.

CAPABILITIES:
✅ Search candidates by skills, location, experience, certifications
✅ List all candidates in database  
✅ Get detailed information about specific candidates
✅ Compare two candidates side-by-side
✅ Provide database statistics

SEARCH EXAMPLES:
- "Find Python developers" → Use Advanced Search
- "Python developers in Phoenix with 5+ years" → Use Advanced Search
- "Who has AWS certification?" → Use Advanced Search with "AWS certified"
- "List all candidates" → Use List All Resumes
- "Tell me about John Doe" → Use Get Candidate Details
- "Compare John and Jane" → Use Compare Two Candidates
- "How many resumes?" → Use Get Resume Count

Be conversational, friendly, and helpful. Understand user intent even with vague queries.""",
    llm=llm,
    tools=[
        advanced_search_tool,
        get_resume_count_tool,
        list_all_resumes_tool,
        get_candidate_details_tool,
        compare_candidates_tool
    ],
    verbose=True,
    max_iter=5,
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


# ============== API ENDPOINTS ==============

@app.get("/")
async def root():
    return {"message": "Master AI Resume Intelligence Chatbot", "status": "running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Master chatbot endpoint - handles ALL user queries"""
    
    print(f"\n{'='*60}")
    print(f"💬 USER: {request.message}")
    print(f"{'='*60}\n")
    
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Create task for Master Chat Agent
    chat_task = Task(
        description=f"""User says: "{request.message}"

Respond to the user's request using your available tools. Be conversational and helpful.

If they're searching for candidates, use the Advanced Search tool.
If they want to list all, use List All Resumes.
If they mention a specific name, use Get Candidate Details or Compare.
If they ask for stats, use Get Resume Count.

Provide a natural, friendly response.""",
        expected_output="Helpful response to user's query",
        agent=master_chat_agent
    )
    
    # Execute
    chat_crew = Crew(
        agents=[master_chat_agent],
        tasks=[chat_task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = chat_crew.kickoff()
        response_text = str(result)
        
        print(f"\n✅ RESPONSE:\n{response_text}\n")
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            agent_used="Master Agent"
        )
    
    except Exception as e:
        error_message = f"I apologize, I encountered an error: {str(e)}. Please try rephrasing."
        return ChatResponse(
            response=error_message,
            conversation_id=conversation_id,
            agent_used="Error Handler"
        )


@app.post("/upload")
async def upload_resumes(files: List[UploadFile] = File(...)):
    """Upload and process resume files using SINGLE master agent"""
    
    processed_resumes = []
    points = []
    
    for file in files:
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
        
        # Extract info using SINGLE master agent
        resume_info = extract_resume_info_with_single_agent(text, file.filename)
        resume_info['id'] = file_id
        
        # Create rich embedding
        embedding_text = create_rich_embedding_text(resume_info)
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
    print("🚀 Starting Master AI Resume Intelligence Chatbot...")
    print(f"📊 Qdrant: {QDRANT_URL}")
    print(f"🤖 Ollama Model: {OLLAMA_MODEL}")
    print(f"📢 Embedding Model: {EMBEDDING_MODEL}")
    print(f"📁 Upload Directory: {UPLOAD_DIR}")
    print(f"📏 Vector Size: {VECTOR_SIZE}")
    print("\n" + "="*60)
    print("💡 SINGLE MASTER AGENT ARCHITECTURE")
    print("="*60)
    print("✅ 1 Extraction Agent - Parses ALL resume data")
    print("✅ 1 Chat Agent - Handles ALL user queries")
    print("✅ No helper functions - Agent does everything!")
    print("✅ Resource efficient - Minimal LLM calls")
    print("="*60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)