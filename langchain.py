from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import os
import uuid
from datetime import datetime, timedelta
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import pickle
from functools import lru_cache
import time

# CrewAI and AI imports
from crewai import Agent, Task, Crew, Process, LLM
from langchain_ollama import OllamaEmbeddings
from crewai.tools import tool

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Document processing
import PyPDF2
import docx

# ============== CONFIGURATION ==============
OLLAMA_MODEL = "mistral:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "resumes_collection"
UPLOAD_DIR = "uploaded_resumes"
CACHE_DIR = "cache"
PROCESSED_CACHE = "processed_resumes.pkl"

# Performance Settings
MAX_WORKERS = 4  # Parallel processing threads
CACHE_TTL_HOURS = 24  # Cache time-to-live
BATCH_SIZE = 10  # Batch processing size
ENABLE_CACHING = True

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ============== INITIALIZE SERVICES ==============
app = FastAPI(title="Enterprise Multi-Agent Resume Intelligence System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_URL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

# Thread pools for parallel processing
extraction_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
embedding_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# LLM Configuration
llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    timeout=1500,
    temperature=0.5,  # Slightly higher for more natural, conversational responses
    max_tokens=3000   # More tokens for detailed explanations
)

# ============== CACHING SYSTEM ==============
class CacheManager:
    def __init__(self):
        self.memory_cache = {}
        self.cache_file = os.path.join(CACHE_DIR, PROCESSED_CACHE)
        self.load_cache()
    
    def load_cache(self):
        """Load cached processed resumes from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.memory_cache = pickle.load(f)
                print(f"✅ Loaded {len(self.memory_cache)} cached resumes")
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}")
            self.memory_cache = {}
    
    def save_cache(self):
        """Save processed resumes to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.memory_cache, f)
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
    
    def get_file_hash(self, content: bytes) -> str:
        """Generate hash for file content"""
        return hashlib.sha256(content).hexdigest()
    
    def get_cached_resume(self, file_hash: str) -> Optional[Dict]:
        """Get cached resume data"""
        if not ENABLE_CACHING:
            return None
        
        cached = self.memory_cache.get(file_hash)
        if cached:
            # Check if cache is still valid
            cache_time = cached.get('cached_at')
            if cache_time:
                age = datetime.now() - datetime.fromisoformat(cache_time)
                if age < timedelta(hours=CACHE_TTL_HOURS):
                    print(f"✅ Cache hit: {cached.get('name', 'Unknown')}")
                    return cached
        return None
    
    def cache_resume(self, file_hash: str, resume_data: Dict):
        """Cache processed resume"""
        if ENABLE_CACHING:
            resume_data['cached_at'] = datetime.now().isoformat()
            resume_data['file_hash'] = file_hash
            self.memory_cache[file_hash] = resume_data
            self.save_cache()

cache_manager = CacheManager()

# ============== QDRANT INITIALIZATION ==============
def initialize_qdrant():
    """Initialize or reuse existing Qdrant collection"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if collection_exists:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            print(f"✅ Reusing existing collection with {collection_info.points_count} resumes")
            return collection_info.vectors_count
        else:
            # Create new collection
            test_emb = embeddings.embed_query("test")
            VECTOR_SIZE = len(test_emb)
            
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"✅ Created new Qdrant collection (vector size: {VECTOR_SIZE})")
            return VECTOR_SIZE
    except Exception as e:
        print(f"❌ Qdrant initialization error: {e}")
        raise

VECTOR_SIZE = initialize_qdrant()

# ============== PYDANTIC MODELS ==============
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    agent_flow: List[str]
    metadata: Optional[Dict[str, Any]] = None

class UploadStatus(BaseModel):
    status: str
    total_files: int
    processed: int
    cached: int
    failed: int
    resumes: List[Dict[str, Any]]
    processing_time: float

# ============== SPECIALIZED TOOLS ==============
@tool("Semantic Search Candidates")
def semantic_search_tool(query: str, filters: str = "{}") -> str:
    """
    Search candidates using semantic understanding with intelligent ranking.
    
    Args:
        query: Natural language search query
        filters: JSON string with filters (optional, defaults to empty object)
    """
    try:
        # Robust filter handling
        if filters is None:
            filters = "{}"
        if not isinstance(filters, str):
            filters = "{}"
        if filters.strip() == "":
            filters = "{}"
            
        print(f"🔍 Searching for: '{query}' with filters: {filters}")
        
        # Parse filters safely
        try:
            filter_dict = json.loads(filters)
        except (json.JSONDecodeError, TypeError):
            filter_dict = {}
        
        # Generate embedding for query
        query_embedding = embeddings.embed_query(query)
        
        # Search Qdrant with higher limit for better filtering
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=50
        )
        
        if not results:
            return "No matching candidates found."
        
        print(f"📊 Found {len(results)} initial results")
        
        # Apply filters and boost scoring
        filtered_results = []
        for result in results:
            resume = result.payload
            score = result.score
            
            # Boost score based on query-role matching
            query_lower = query.lower()
            role_lower = resume.get('current_role', '').lower()
            
            # Exact role match gets highest boost
            if query_lower in role_lower or role_lower in query_lower:
                score = score * 1.5
            
            # Check for key terms in query matching skills
            query_terms = query_lower.split()
            resume_skills = [s.lower() for s in resume.get('skills', [])]
            matching_skills = sum(1 for term in query_terms if any(term in skill for skill in resume_skills))
            if matching_skills > 0:
                score = score * (1 + matching_skills * 0.1)
            
            # Location filter
            if filter_dict.get("location"):
                if filter_dict["location"].lower() not in resume.get('location', '').lower():
                    continue
            
            # Experience filter
            if filter_dict.get("min_years"):
                if resume.get('years_experience', 0) < filter_dict["min_years"]:
                    continue
            
            # Skills filter
            if filter_dict.get("required_skills"):
                required = [s.lower() for s in filter_dict["required_skills"]]
                if not any(skill in resume_skills for skill in required):
                    continue
            
            # Store boosted score
            filtered_results.append((result, score))
        
        if not filtered_results:
            return "No candidates match the specified filters."
        
        # Sort by boosted score
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ After filtering and ranking: {len(filtered_results)} candidates")
        
        # Format results with rich context for LLM analysis
        output = f"SEARCH RESULTS for '{query}':\n"
        output += f"Total matches: {len(filtered_results)}\n\n"
        
        for i, (result, boosted_score) in enumerate(filtered_results[:5], 1):
            resume = result.payload
            output += f"CANDIDATE #{i}:\n"
            output += f"Name: {resume.get('name', 'N/A')}\n"
            output += f"Role: {resume.get('current_role', 'N/A')}\n"
            output += f"Experience: {resume.get('years_experience', 0)} years\n"
            output += f"Location: {resume.get('location', 'N/A')}\n"
            output += f"Email: {resume.get('email', 'N/A')}\n"
            output += f"Skills: {', '.join(resume.get('skills', []))}\n"
            
            if resume.get('certifications'):
                output += f"Certifications: {', '.join(resume.get('certifications', []))}\n"
            
            if resume.get('previous_companies'):
                output += f"Previous Companies: {', '.join(resume.get('previous_companies', []))}\n"
            
            output += f"Relevance Score: {boosted_score*100:.1f}%\n"
            output += f"---\n\n"
        
        return output
        
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg


@tool("Get Candidate Details")
def get_candidate_details_tool(name: str) -> str:
    """Retrieve detailed information about a specific candidate"""
    try:
        results = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=100)
        for point in results[0]:
            resume = point.payload
            if name.lower() in resume.get('name', '').lower():
                details = f"{'='*60}\n"
                details += f"👤 {resume.get('name', 'N/A')}\n"
                details += f"{'='*60}\n"
                details += f"📧 Email: {resume.get('email', 'N/A')}\n"
                details += f"📱 Phone: {resume.get('phone', 'N/A')}\n"
                details += f"📍 Location: {resume.get('location', 'N/A')}\n"
                details += f"💼 Role: {resume.get('current_role', 'N/A')}\n"
                details += f"📅 Experience: {resume.get('years_experience', 0)} years\n"
                details += f"💰 Salary: {resume.get('salary_expectations', 'N/A')}\n\n"
                details += f"🛠️ SKILLS:\n{', '.join(resume.get('skills', []))}\n\n"
                if resume.get('certifications'):
                    details += f"🏆 CERTS:\n{', '.join(resume.get('certifications', []))}\n\n"
                details += f"🎓 EDUCATION:\n{resume.get('degree', 'N/A')}\n"
                details += f"{resume.get('university', 'N/A')} ({resume.get('graduation_year', 'N/A')})\n"
                return details
        return f"Candidate '{name}' not found."
    except Exception as e:
        return f"Error: {str(e)}"

@tool("Compare Candidates")
def compare_candidates_tool(name1: str, name2: str) -> str:
    """Compare two candidates side-by-side"""
    try:
        results = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=100)
        c1 = c2 = None
        
        for point in results[0]:
            resume = point.payload
            name = resume.get('name', '').lower()
            if name1.lower() in name and not c1:
                c1 = resume
            if name2.lower() in name and not c2:
                c2 = resume
            if c1 and c2:
                break
        
        if not c1 or not c2:
            missing = []
            if not c1: missing.append(name1)
            if not c2: missing.append(name2)
            return f"Could not find: {', '.join(missing)}"
        
        comparison = f"""
{'='*70}
CANDIDATE COMPARISON
{'='*70}
{c1['name']:35} | {c2['name']}
{'-'*70}
📧 {c1.get('email', 'N/A'):35} | {c2.get('email', 'N/A')}
📍 {c1.get('location', 'N/A'):35} | {c2.get('location', 'N/A')}
💼 {c1.get('current_role', 'N/A'):35} | {c2.get('current_role', 'N/A')}
📅 {c1.get('years_experience', 0)} years{' '*28} | {c2.get('years_experience', 0)} years

🛠️ SKILLS:
C1: {', '.join(c1.get('skills', []))}
C2: {', '.join(c2.get('skills', []))}

Common: {', '.join(set(c1.get('skills', [])) & set(c2.get('skills', [])))}
"""
        return comparison
    except Exception as e:
        return f"Error: {str(e)}"

@tool("Get Database Statistics")
def get_database_stats_tool() -> str:
    """Get comprehensive database statistics"""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        count = collection_info.points_count
        results = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=100)
        
        total_exp = 0
        locations = {}
        skills_count = {}
        
        for point in results[0]:
            resume = point.payload
            total_exp += resume.get('years_experience', 0)
            loc = resume.get('location', 'Unknown')
            locations[loc] = locations.get(loc, 0) + 1
            for skill in resume.get('skills', []):
                skills_count[skill] = skills_count.get(skill, 0) + 1
        
        avg_exp = total_exp / count if count > 0 else 0
        top_skills = sorted(skills_count.items(), key=lambda x: x[1], reverse=True)[:5]
        top_locations = sorted(locations.items(), key=lambda x: x[1], reverse=True)[:3]
        
        stats = f"""
📊 DATABASE STATISTICS
{'='*50}
Total Resumes: {count}
Average Experience: {avg_exp:.1f} years
Cached Resumes: {len(cache_manager.memory_cache)}

📍 Top Locations:
{chr(10).join([f"   {loc}: {cnt}" for loc, cnt in top_locations])}

🛠️ Top Skills:
{chr(10).join([f"   {skill}: {cnt}" for skill, cnt in top_skills])}
"""
        return stats
    except Exception as e:
        return f"Error: {str(e)}"

@tool("List All Candidates")
def list_all_candidates_tool() -> str:
    """List all candidates with summary"""
    try:
        results = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=50)
        if not results[0]:
            return "No resumes in database."
        
        output = f"📋 All Candidates ({len(results[0])}):\n\n"
        for i, point in enumerate(results[0], 1):
            resume = point.payload
            output += f"{i}. {resume.get('name', 'N/A')} - {resume.get('current_role', 'N/A')}\n"
            output += f"   📍 {resume.get('location', 'N/A')} | {resume.get('years_experience', 0)}y\n"
            output += f"   🛠️ {', '.join(resume.get('skills', [])[:5])}\n\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

# ============== SPECIALIZED AGENTS ==============
extraction_specialist = Agent(
    role="Resume Data Extraction Specialist",
    goal="Extract comprehensive structured data from resume text with high accuracy",
    backstory="""You are an expert at parsing resumes and extracting structured information.
    You excel at identifying key details even in poorly formatted resumes.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

search_specialist = Agent(
    role="Semantic Search Specialist",
    goal="Find relevant candidates using advanced search techniques",
    backstory="""Expert at understanding search queries and semantic matching.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

comparison_specialist = Agent(
    role="Candidate Comparison Analyst",
    goal="Compare candidates and provide detailed analysis",
    backstory="""Expert at analyzing and comparing candidates side-by-side.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

master_orchestrator = Agent(
    role="AI Resume Intelligence Assistant",
    goal="Provide intelligent, conversational analysis of resumes and candidates",
    backstory="""You are an expert AI recruiter and resume analyst. You help users find and understand candidates.

When analyzing search results:
- Explain WHY candidates are good matches in natural language
- Highlight specific skills and experience that matter
- Compare candidates' strengths
- Make clear recommendations
- Be conversational and helpful

IMPORTANT: You must provide your analysis in natural, conversational language - not just internal thoughts.
Write your response as if you're talking directly to a recruiter.""",
    llm=llm,
    verbose=True,
    max_iter=3,
    allow_delegation=False,
    tools=[
        semantic_search_tool,
        get_candidate_details_tool,
        compare_candidates_tool,
        get_database_stats_tool,
        list_all_candidates_tool
    ]
)

# ============== EXTRACTION FUNCTIONS ==============
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX"""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error: {str(e)}"

def extract_years(value) -> int:
    """Extract numeric years from various formats"""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r'(\d+)', str(value))
        if match:
            return int(match.group(1))
    return 0

def ensure_list(value) -> List:
    """Ensure value is a list"""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value] if value and value != "Not found" else []
    return []

def process_resume_extraction(text: str, filename: str) -> dict:
    """Process resume with improved error handling"""
    print(f"\n📄 Processing: {filename}")
    
    try:
        task = Task(
            description=f"""Extract information from resume and return ONLY valid JSON:

{text[:2500]}

Return format (no markdown, no extra text):
{{
  "name": "First Last",
  "email": "email@domain.com",
  "phone": "123-456-7890",
  "location": "City, State",
  "linkedin": "URL or Not found",
  "portfolio": "URL or Not found",
  "current_role": "Job Title",
  "years_experience": 5,
  "previous_companies": ["Company1"],
  "salary_expectations": "$120K or Not found",
  "skills": ["Python", "AWS"],
  "certifications": ["AWS Certified"],
  "degree": "BS Computer Science",
  "university": "University Name",
  "graduation_year": "2020"
}}

CRITICAL: years_experience must be NUMBER only, not "5 years".""",
            expected_output="Valid JSON",
            agent=extraction_specialist
        )
        
        crew = Crew(
            agents=[extraction_specialist],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        raw_output = str(result.tasks_output[0].raw) if hasattr(result, 'tasks_output') else str(result)
        
        # Clean output
        raw_output = raw_output.strip()
        if raw_output.startswith('```'):
            raw_output = re.sub(r'^```json?\s*', '', raw_output)
            raw_output = re.sub(r'\s*```$', '', raw_output)
        
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            json_str = json_match.group(0)
            json_str = json_str.replace('\n', ' ')
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            extracted_data = json.loads(json_str)
        else:
            raise ValueError("No JSON found")
        
        resume_info = {
            "filename": filename,
            "name": str(extracted_data.get("name", "Not found")),
            "email": str(extracted_data.get("email", "Not found")),
            "phone": str(extracted_data.get("phone", "Not found")),
            "location": str(extracted_data.get("location", "Not found")),
            "linkedin": str(extracted_data.get("linkedin", "Not found")),
            "portfolio": str(extracted_data.get("portfolio", "Not found")),
            "current_role": str(extracted_data.get("current_role", "Not specified")),
            "years_experience": extract_years(extracted_data.get("years_experience", 0)),
            "previous_companies": ensure_list(extracted_data.get("previous_companies", []))[:5],
            "salary_expectations": str(extracted_data.get("salary_expectations", "Not specified")),
            "skills": ensure_list(extracted_data.get("skills", []))[:15],
            "certifications": ensure_list(extracted_data.get("certifications", []))[:10],
            "degree": str(extracted_data.get("degree", "Not found")),
            "university": str(extracted_data.get("university", "Not found")),
            "graduation_year": str(extracted_data.get("graduation_year", "Not found")),
            "uploaded_at": datetime.now().isoformat()
        }
        
        print(f"✅ {resume_info['name']} - {resume_info['current_role']}")
        return resume_info
        
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
        return {
            "filename": filename,
            "name": f"Failed - {filename}",
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

# ============== PARALLEL PROCESSING ==============
async def process_single_resume(file_content: bytes, filename: str, file_extension: str) -> Optional[Dict]:
    """Process a single resume with caching"""
    try:
        # Check cache first
        file_hash = cache_manager.get_file_hash(file_content)
        cached_resume = cache_manager.get_cached_resume(file_hash)
        
        if cached_resume:
            return {
                'resume': cached_resume,
                'cached': True,
                'file_id': cached_resume.get('id', str(uuid.uuid4()))
            }
        
        # Save file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Extract text
        if file_extension == '.pdf':
            text = await asyncio.get_event_loop().run_in_executor(
                extraction_executor, extract_text_from_pdf, file_path
            )
        elif file_extension == '.docx':
            text = await asyncio.get_event_loop().run_in_executor(
                extraction_executor, extract_text_from_docx, file_path
            )
        else:
            return None
        
        if "Error" in text:
            return None
        
        # Process with AI
        resume_info = await asyncio.get_event_loop().run_in_executor(
            extraction_executor, process_resume_extraction, text, filename
        )
        resume_info['id'] = file_id
        
        # Cache the result
        cache_manager.cache_resume(file_hash, resume_info)
        
        return {
            'resume': resume_info,
            'cached': False,
            'file_id': file_id
        }
        
    except Exception as e:
        print(f"❌ Processing error for {filename}: {e}")
        return None

async def process_resumes_parallel(files: List[UploadFile]) -> UploadStatus:
    """Process multiple resumes in parallel with intelligent batching"""
    start_time = time.time()
    
    # Read all files first
    file_data = []
    for file in files:
        content = await file.read()
        extension = os.path.splitext(file.filename)[1]
        file_data.append((content, file.filename, extension))
    
    # Process in parallel
    tasks = [
        process_single_resume(content, filename, ext)
        for content, filename, ext in file_data
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Separate results
    processed_resumes = []
    cached_count = 0
    failed_count = 0
    points = []
    
    for result in results:
        if result is None:
            failed_count += 1
            continue
        
        resume_info = result['resume']
        if result['cached']:
            cached_count += 1
        
        processed_resumes.append(resume_info)
        
        # Create rich embedding with weighted fields
        # Role is most important, then skills, then name/location
        embedding_text = (
            f"ROLE: {resume_info['current_role']} {resume_info['current_role']} "
            f"SKILLS: {' '.join(resume_info['skills'])} "
            f"EXPERIENCE: {resume_info['years_experience']} years "
            f"NAME: {resume_info['name']} "
            f"LOCATION: {resume_info['location']} "
            f"COMPANIES: {' '.join(resume_info['previous_companies'])}"
        )
        embedding = embeddings.embed_query(embedding_text)
        
        point = PointStruct(
            id=result['file_id'],
            vector=embedding,
            payload=resume_info
        )
        points.append(point)
    
    # Batch upsert to Qdrant
    if points:
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i+BATCH_SIZE]
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)
    
    processing_time = time.time() - start_time
    
    return UploadStatus(
        status="success",
        total_files=len(files),
        processed=len(processed_resumes),
        cached=cached_count,
        failed=failed_count,
        resumes=processed_resumes,
        processing_time=round(processing_time, 2)
    )

def process_user_query(message: str, conversation_id: str) -> Dict[str, Any]:
    """Master orchestration function that routes queries to appropriate agents"""
    print(f"\n{'='*70}")
    print(f"🎯 ORCHESTRATOR: Processing query")
    print(f"💬 User: {message}")
    print(f"{'='*70}\n")
    
    agent_flow = ["Master Orchestrator"]
    
    # Create task with focus on single tool call + analysis
    orchestrator_task = Task(
        description=f"""User query: "{message}"

Step 1: Call the appropriate tool ONCE to get data
Step 2: Analyze the tool output and provide intelligent insights

For search queries:
- Call Semantic Search Candidates with: {{"query": "search terms", "filters": "{{}}"}}
- The tool returns complete candidate details with rankings
- Analyze the results you received
- DO NOT call additional tools - all data is already in the search results
- Explain why candidates match, highlight their strengths, make recommendations

For other query types, use the appropriate tool once and analyze.

Provide an intelligent, conversational response that explains your reasoning.""",
        expected_output="Intelligent analysis with clear explanations and recommendations",
        agent=master_orchestrator
    )
    
    # Execute orchestration
    crew = Crew(
        agents=[master_orchestrator],
        tasks=[orchestrator_task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        result = crew.kickoff()
        response_text = str(result)
        
        print(f"\n✅ ORCHESTRATOR COMPLETE")
        print(f"📤 Response: {response_text[:200]}...")
        print(f"📊 Agent Flow: {' → '.join(agent_flow)}\n")
        
        return {
            "response": response_text,
            "agent_flow": agent_flow,
            "status": "success"
        }
        
    except Exception as e:
        print(f"\n❌ ORCHESTRATION ERROR: {str(e)}\n")
        return {
            "response": f"I apologize, I encountered an error: {str(e)}. Please try rephrasing your question.",
            "agent_flow": agent_flow + ["Error Handler"],
            "status": "error"
        }


# ============== API ENDPOINTS ==============
@app.get("/")
async def root():
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    return {
        "message": "Enterprise Multi-Agent Resume Intelligence System",
        "status": "operational",
        "features": [
            "Intelligent Caching",
            "Parallel Processing",
            "Semantic Search",
            "Multi-Agent Orchestration"
        ],
        "stats": {
            "total_resumes": collection_info.points_count,
            "cached_resumes": len(cache_manager.memory_cache),
            "max_workers": MAX_WORKERS
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with intelligent orchestration"""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    result = process_user_query(request.message, conversation_id)
    
    return ChatResponse(
        response=result["response"],
        conversation_id=conversation_id,
        agent_flow=result["agent_flow"],
        metadata={"status": result["status"]}
    )

@app.post("/upload", response_model=UploadStatus)
async def upload_resumes(files: List[UploadFile] = File(...)):
    """Upload with parallel processing and caching"""
    print(f"\n{'='*70}")
    print(f"📤 Upload Request: {len(files)} files")
    print(f"{'='*70}\n")
    
    result = await process_resumes_parallel(files)
    
    print(f"\n{'='*70}")
    print(f"✅ Upload Complete")
    print(f"   Total: {result.total_files}")
    print(f"   Processed: {result.processed}")
    print(f"   Cached: {result.cached}")
    print(f"   Failed: {result.failed}")
    print(f"   Time: {result.processing_time}s")
    print(f"{'='*70}\n")
    
    return result

@app.get("/resumes")
async def get_all_resumes():
    """Get all stored resumes"""
    try:
        results = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=100)
        resumes = []
        for point in results[0]:
            resume_data = point.payload
            resume_data['id'] = str(point.id)
            resumes.append(resume_data)
        return {
            "resumes": resumes,
            "count": len(resumes),
            "cached_count": len(cache_manager.memory_cache)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics"""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        
        # Calculate cache hit rate
        total_cached = len(cache_manager.memory_cache)
        cache_hit_rate = (total_cached / collection_info.points_count * 100) if collection_info.points_count > 0 else 0
        
        return {
            "total_resumes": collection_info.points_count,
            "cached_resumes": total_cached,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "collection": COLLECTION_NAME,
            "vector_size": VECTOR_SIZE,
            "max_workers": MAX_WORKERS,
            "cache_enabled": ENABLE_CACHING,
            "cache_ttl_hours": CACHE_TTL_HOURS,
            "architecture": "Multi-Agent Orchestrator with Parallel Processing",
            "status": "operational"
        }
    except Exception as e:
        return {
            "total_resumes": 0,
            "status": "error",
            "message": str(e)
        }

@app.delete("/resumes/clear")
async def clear_all_resumes():
    """Clear all resumes and cache"""
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        
        # Clear cache
        cache_manager.memory_cache = {}
        cache_manager.save_cache()
        
        return {
            "message": "All resumes and cache cleared successfully",
            "resumes_cleared": True,
            "cache_cleared": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache():
    """Clear only the cache (keep resumes)"""
    try:
        old_count = len(cache_manager.memory_cache)
        cache_manager.memory_cache = {}
        cache_manager.save_cache()
        
        return {
            "message": f"Cache cleared: {old_count} entries removed",
            "cleared_entries": old_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/stats")
async def get_cache_stats():
    """Get detailed cache statistics"""
    try:
        cache_size = len(cache_manager.memory_cache)
        
        # Calculate cache ages
        ages = []
        for cached_data in cache_manager.memory_cache.values():
            if cached_data.get('cached_at'):
                cache_time = datetime.fromisoformat(cached_data['cached_at'])
                age = datetime.now() - cache_time
                ages.append(age.total_seconds() / 3600)  # hours
        
        avg_age = sum(ages) / len(ages) if ages else 0
        
        return {
            "total_cached": cache_size,
            "average_age_hours": round(avg_age, 2),
            "oldest_cache_hours": round(max(ages), 2) if ages else 0,
            "newest_cache_hours": round(min(ages), 2) if ages else 0,
            "cache_file": PROCESSED_CACHE,
            "cache_ttl_hours": CACHE_TTL_HOURS
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resumes/reprocess")
async def reprocess_failed_resumes():
    """Reprocess resumes that failed extraction"""
    try:
        results = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=100)
        failed_resumes = []
        
        for point in results[0]:
            resume = point.payload
            if resume.get('name', '').startswith('Failed -') or resume.get('name') == 'Extraction Failed':
                failed_resumes.append({
                    'id': str(point.id),
                    'filename': resume.get('filename')
                })
        
        if not failed_resumes:
            return {
                "message": "No failed resumes to reprocess",
                "count": 0
            }
        
        return {
            "message": f"Found {len(failed_resumes)} failed resumes",
            "failed_resumes": failed_resumes,
            "action": "Please re-upload these files to reprocess"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check Qdrant
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        health_status["components"]["qdrant"] = "operational"
    except Exception as e:
        health_status["components"]["qdrant"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check LLM
    try:
        test_agent = Agent(
            role="Test",
            goal="Test",
            backstory="Test",
            llm=llm,
            verbose=False
        )
        health_status["components"]["llm"] = "operational"
    except Exception as e:
        health_status["components"]["llm"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Embeddings
    try:
        embeddings.embed_query("test")
        health_status["components"]["embeddings"] = "operational"
    except Exception as e:
        health_status["components"]["embeddings"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Cache
    try:
        cache_count = len(cache_manager.memory_cache)
        health_status["components"]["cache"] = f"operational ({cache_count} entries)"
    except Exception as e:
        health_status["components"]["cache"] = f"error: {str(e)}"
    
    return health_status

@app.get("/search/advanced")
async def advanced_search(
    query: str,
    location: Optional[str] = None,
    min_years: Optional[int] = None,
    skills: Optional[str] = None,
    limit: int = 10
):
    """Advanced search with URL parameters"""
    try:
        query_embedding = embeddings.embed_query(query)
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=50
        )
        
        filtered = []
        for result in results:
            resume = result.payload
            
            # Apply filters
            if location and location.lower() not in resume.get('location', '').lower():
                continue
            if min_years and resume.get('years_experience', 0) < min_years:
                continue
            if skills:
                required_skills = [s.strip().lower() for s in skills.split(',')]
                resume_skills = [s.lower() for s in resume.get('skills', [])]
                if not any(skill in resume_skills for skill in required_skills):
                    continue
            
            filtered.append({
                "resume": resume,
                "score": result.score
            })
        
        return {
            "query": query,
            "filters": {
                "location": location,
                "min_years": min_years,
                "skills": skills
            },
            "total_results": len(filtered),
            "results": filtered[:limit]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Run on startup"""
    print("\n" + "="*70)
    print("🚀 ENTERPRISE MULTI-AGENT RESUME INTELLIGENCE SYSTEM")
    print("="*70)
    print("🎯 Master Orchestrator Agent")
    print("   ├── 📄 Extraction Specialist (Resume Parsing)")
    print("   ├── 🔍 Search Specialist (Semantic Search)")
    print("   ├── 📊 Comparison Specialist (Candidate Analysis)")
    print("   └── 💾 Database Specialist (Stats & Listings)")
    print("="*70)
    print(f"📊 Qdrant: {QDRANT_URL}")
    print(f"🤖 LLM: {OLLAMA_MODEL}")
    print(f"🔢 Vector Size: {VECTOR_SIZE}")
    print(f"🌐 Server: http://127.0.0.1:8000")
    print(f"⚡ Workers: {MAX_WORKERS}")
    print(f"💾 Cache: {'Enabled' if ENABLE_CACHING else 'Disabled'} ({len(cache_manager.memory_cache)} entries)")
    print("="*70)
    print("\n✅ PRODUCTION FEATURES:")
    print("   • Intelligent Caching System (24hr TTL)")
    print("   • Parallel Resume Processing")
    print("   • Batch Vector Embeddings")
    print("   • Multi-Agent Orchestration")
    print("   • Advanced Search API")
    print("   • Health Monitoring")
    print("   • Error Recovery")
    print("   • Performance Optimization")
    print("\n" + "="*70)
    print("📚 API ENDPOINTS:")
    print("   GET  /              - System overview")
    print("   POST /chat          - AI chat interface")
    print("   POST /upload        - Upload resumes (parallel)")
    print("   GET  /resumes       - List all resumes")
    print("   GET  /stats         - System statistics")
    print("   GET  /health        - Health check")
    print("   GET  /search/advanced - Advanced search")
    print("   GET  /cache/stats   - Cache statistics")
    print("   POST /cache/clear   - Clear cache")
    print("   DELETE /resumes/clear - Clear all data")
    print("="*70 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown"""
    print("\n" + "="*70)
    print("🛑 SHUTTING DOWN")
    print("="*70)
    
    # Save cache before shutdown
    cache_manager.save_cache()
    print(f"💾 Cache saved: {len(cache_manager.memory_cache)} entries")
    
    # Cleanup thread pools
    extraction_executor.shutdown(wait=True)
    embedding_executor.shutdown(wait=True)
    print("⚡ Thread pools closed")
    
    print("="*70)
    print("✅ Shutdown complete")
    print("="*70 + "\n")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=True
    )