"""
Arytic - Intelligent Resume RAG System
Version 6.0 - LLM-driven extraction with intelligent matching
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, TypedDict, Annotated, Sequence
import uvicorn
import os
import uuid
from datetime import datetime
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import operator

# LangGraph and LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, END

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Document processing
import PyPDF2
import docx

# ============== CONFIGURATION ==============
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "arytic_resumes"
UPLOAD_DIR = "arytic_uploads"

# Optimized CPU settings
CPU_THREADS = 4
MAX_CONCURRENT_UPLOADS = 3
MAX_WORKERS = 3
CONTEXT_SIZE = 2048

# Processing limits
MAX_SEARCH_RESULTS = 30
TOP_CANDIDATES = 5

os.makedirs(UPLOAD_DIR, exist_ok=True)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ============== FASTAPI SETUP ==============
app = FastAPI(title="Arytic - Intelligent Resume RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== INITIALIZE SERVICES ==============
qdrant_client = QdrantClient(url=QDRANT_URL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

llm = ChatOllama(
    model=OLLAMA_MODEL, 
    base_url=OLLAMA_BASE_URL, 
    temperature=0.1,
    num_ctx=CONTEXT_SIZE,
    num_thread=CPU_THREADS,
    num_gpu=0
)

def initialize_qdrant():
    try:
        collections = qdrant_client.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            test_emb = embeddings.embed_query("test")
            VECTOR_SIZE = len(test_emb)
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"✅ Created Arytic collection (vector size: {VECTOR_SIZE})")
        else:
            print(f"✅ Using existing Arytic collection")
    except Exception as e:
        print(f"❌ Qdrant error: {e}")

initialize_qdrant()

# ============== PYDANTIC MODELS ==============
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    execution_details: Dict[str, Any]

class UploadResponse(BaseModel):
    status: str
    processed: int
    resumes: List[Dict[str, Any]]

# ============== LANGGRAPH STATE ==============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    parsed_requirements: Dict[str, Any]
    search_results: List[Dict]
    analyzed_candidates: List[Dict]
    final_recommendation: str
    next_agent: str
    execution_log: List[str]

# ============== INTELLIGENT EXTRACTION AGENT ==============

class IntelligentExtractionAgent:
    """LLM-driven extraction - learns patterns from resume structure"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "🤖 Intelligent Extraction Agent"
    
    def extract_resume_data(self, text: str, filename: str) -> dict:
        """Use LLM to intelligently extract all resume fields"""
        
        print(f"\n   🤖 {self.name}: Analyzing resume")
        
        # Truncate text to fit context
        resume_text = text[:4000]
        
        extraction_prompt = f"""You are an expert resume parser. Extract information from this resume text.

RESUME TEXT:
{resume_text}

Extract the following information and return as JSON:

{{
  "name": "candidate's full name from header",
  "email": "email address (format: user@domain.com)",
  "phone": "phone number with proper formatting",
  "location": "current location or city/state mentioned",
  "current_role": "most recent job title",
  "years_experience": "total years of professional experience (number only)",
  "skills": ["list all technical and professional skills mentioned"],
  "certifications": ["list all certifications, licenses, or credentials"],
  "education": {{
    "degree": "highest degree earned",
    "university": "university or institution name",
    "graduation_year": "year of graduation"
  }},
  "experience": [
    {{
      "company": "company name",
      "role": "job title",
      "duration": "time period (e.g., 2020-2023)",
      "description": "brief summary of responsibilities"
    }}
  ]
}}

EXTRACTION INSTRUCTIONS:
1. NAME: Look at the top of resume, usually in large text
2. EMAIL: Find pattern like name@company.com
3. PHONE: Find 10-digit phone number patterns
4. LOCATION: Look for City, State or address information
5. CURRENT ROLE: First job title listed in experience section
6. YEARS OF EXPERIENCE: 
   - Look for phrases like "10+ years experience"
   - OR calculate from employment dates (2015-present = 8-9 years)
   - OR count years across all jobs
7. SKILLS: 
   - Look in "Skills" or "Technical Skills" section
   - Include programming languages, tools, technologies
   - Include soft skills if clearly listed
8. CERTIFICATIONS:
   - Look in "Certifications" or "Licenses" section
   - Include professional certifications (PMP, AWS, Scrum, etc.)
   - Include licenses (CPA, PE, etc.)
9. EDUCATION:
   - Find degree name (Bachelor's, Master's, PhD, etc.)
   - Find university name
   - Find graduation year
10. EXPERIENCE:
    - Extract last 3-4 jobs
    - Include company, role, duration, brief description

CRITICAL RULES:
- If you cannot find something, use "N/A" for strings, 0 for numbers, or [] for arrays
- Be precise - only extract what is clearly stated
- For skills and certifications, look for dedicated sections first
- Do not infer or assume information
- Return ONLY valid JSON, no explanations

JSON:"""

        try:
            result = self.llm.invoke(extraction_prompt)
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            # Clean response
            result_text = result_text.strip()
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*', '', result_text)
            
            # Extract JSON
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            
            if json_match:
                json_str = json_match.group(0)
                # Remove trailing commas
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                
                data = json.loads(json_str)
                
                # Validate and structure data
                final_data = {
                    "name": str(data.get('name', 'N/A')).strip(),
                    "email": str(data.get('email', 'N/A')).strip(),
                    "phone": str(data.get('phone', 'N/A')).strip(),
                    "location": str(data.get('location', 'N/A')).strip(),
                    "current_role": str(data.get('current_role', 'N/A')).strip(),
                    "years_experience": self._parse_years(data.get('years_experience', 0)),
                    "skills": self._parse_list(data.get('skills', [])),
                    "certifications": self._parse_list(data.get('certifications', [])),
                    "degree": "N/A",
                    "university": "N/A",
                    "graduation_year": "N/A",
                    "experience": []
                }
                
                # Parse education
                if isinstance(data.get('education'), dict):
                    edu = data['education']
                    final_data['degree'] = str(edu.get('degree', 'N/A')).strip()
                    final_data['university'] = str(edu.get('university', 'N/A')).strip()
                    final_data['graduation_year'] = str(edu.get('graduation_year', 'N/A')).strip()
                
                # Parse experience
                if isinstance(data.get('experience'), list):
                    final_data['experience'] = data['experience'][:5]
                
                # Create companies list from experience
                final_data['previous_companies'] = [
                    exp.get('company', 'N/A') 
                    for exp in final_data['experience'] 
                    if exp.get('company')
                ][:5]
                
                print(f"   ✅ {final_data['name']} | {final_data['current_role']}")
                print(f"   ✅ Skills: {len(final_data['skills'])} | Certs: {len(final_data['certifications'])}")
                print(f"   ✅ Location: {final_data['location']} | Experience: {final_data['years_experience']} years")
                
                return final_data
            else:
                print(f"   ⚠️ Could not parse JSON, using fallback")
                return self.create_fallback_data(text, filename)
                
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON parse error: {e}")
            return self.create_fallback_data(text, filename)
        except Exception as e:
            print(f"   ⚠️ Extraction error: {e}")
            return self.create_fallback_data(text, filename)
    
    def _parse_years(self, value) -> int:
        """Parse years experience"""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            # Extract number from string
            match = re.search(r'(\d+)', value)
            if match:
                return int(match.group(1))
        return 0
    
    def _parse_list(self, value) -> list:
        """Parse list fields"""
        if isinstance(value, list):
            return [str(item).strip() for item in value if item][:30]
        return []
    
    def create_fallback_data(self, text: str, filename: str) -> dict:
        """Fallback extraction using simple patterns"""
        print(f"   🔄 Using fallback extraction")
        
        # Simple regex fallbacks
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        
        name = filename.replace('.pdf', '').replace('.docx', '').replace('_', ' ').strip().title()
        
        return {
            "name": name,
            "email": email_match.group(0) if email_match else "N/A",
            "phone": phone_match.group(0) if phone_match else "N/A",
            "location": "N/A",
            "current_role": "N/A",
            "years_experience": 0,
            "skills": [],
            "certifications": [],
            "degree": "N/A",
            "university": "N/A",
            "graduation_year": "N/A",
            "experience": [],
            "previous_companies": []
        }


# ============== AGENT 1: QUERY UNDERSTANDING ==============
class QueryUnderstandingAgent:
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "🧠 Query Understanding Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        
        log_msg = f"{self.name}: Analyzing query"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        understanding_prompt = f"""Analyze this hiring query and extract key requirements.

QUERY: "{query}"

Extract requirements and return JSON:
{{
  "role": "job title if mentioned, else null",
  "skills": ["required skills if mentioned"],
  "experience_years": "number if mentioned, else null",
  "location": "location if mentioned, else null",
  "certifications": ["certifications if mentioned"],
  "keywords": ["other important keywords from query"]
}}

EXAMPLES:
Query: "Find me a software developer with Python experience"
→ {{"role": "software developer", "skills": ["Python"], "experience_years": null, "location": null, "certifications": [], "keywords": []}}

Query: "I need someone with 5 years experience in AWS and based in California"
→ {{"role": null, "skills": ["AWS"], "experience_years": 5, "location": "California", "certifications": [], "keywords": []}}

Query: "Looking for PMP certified project manager"
→ {{"role": "project manager", "skills": [], "experience_years": null, "location": null, "certifications": ["PMP"], "keywords": []}}

RULES:
- Only extract what is explicitly mentioned
- Use null for missing information
- Be precise with skill names
- Return valid JSON only

JSON:"""

        try:
            result = self.llm.invoke(understanding_prompt)
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            # Clean and parse
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*', '', result_text)
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            
            if json_match:
                json_str = json_match.group(0)
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                requirements = json.loads(json_str)
                
                state["parsed_requirements"] = requirements
                
                log_msg = f"✓ Requirements: {requirements}"
                print(f"   {log_msg}")
                state["execution_log"].append(log_msg)
            else:
                state["parsed_requirements"] = {}
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            state["parsed_requirements"] = {}
        
        state["next_agent"] = "search"
        return state


# ============== AGENT 2: INTELLIGENT SEARCH ==============
class IntelligentSearchAgent:
    
    def __init__(self, qdrant_client, embeddings):
        self.qdrant = qdrant_client
        self.embeddings = embeddings
        self.name = "🔍 Intelligent Search Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        requirements = state.get("parsed_requirements", {})
        
        log_msg = f"{self.name}: Searching candidates"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        # Build semantic search query
        search_parts = []
        if requirements.get('role'):
            search_parts.append(requirements['role'])
        if requirements.get('skills'):
            search_parts.extend(requirements['skills'])
        if requirements.get('keywords'):
            search_parts.extend(requirements['keywords'])
        if requirements.get('certifications'):
            search_parts.extend(requirements['certifications'])
        
        search_query = ' '.join(search_parts) if search_parts else query
        
        log_msg = f"Search: {search_query}"
        print(f"   {log_msg}")
        state["execution_log"].append(log_msg)
        
        # Semantic search with embeddings
        query_embedding = self.embeddings.embed_query(search_query)
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=MAX_SEARCH_RESULTS
        )
        
        if not results:
            state["next_agent"] = "END"
            state["final_recommendation"] = "❌ No candidates found. Please upload resumes first."
            return state
        
        candidates = []
        for r in results:
            candidate = r.payload.copy()
            candidate['semantic_score'] = round(r.score * 100, 1)
            candidates.append(candidate)
        
        state["search_results"] = candidates
        
        log_msg = f"✓ Found {len(candidates)} candidates"
        print(f"   {log_msg}")
        state["execution_log"].append(log_msg)
        
        state["next_agent"] = "analysis"
        return state


# ============== AGENT 3: INTELLIGENT MATCHING & RANKING ==============
class IntelligentMatchingAgent:
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "🎯 Intelligent Matching Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        candidates = state["search_results"][:TOP_CANDIDATES]
        requirements = state.get("parsed_requirements", {})
        query = state["user_query"]
        
        log_msg = f"{self.name}: Analyzing top {len(candidates)} candidates"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        # Prepare candidate data
        candidate_summaries = []
        for i, c in enumerate(candidates):
            summary = {
                "id": i,
                "name": c.get('name'),
                "role": c.get('current_role'),
                "experience_years": c.get('years_experience'),
                "skills": c.get('skills', [])[:20],
                "certifications": c.get('certifications', []),
                "location": c.get('location'),
                "degree": c.get('degree'),
                "companies": c.get('previous_companies', [])[:3]
            }
            candidate_summaries.append(summary)
        
        matching_prompt = f"""You are an expert recruiter. Analyze these candidates and rank them by fit.

USER QUERY: "{query}"

REQUIREMENTS:
{json.dumps(requirements, indent=2)}

CANDIDATES:
{json.dumps(candidate_summaries, indent=2)}

For each candidate, provide match analysis:
[
  {{
    "id": 0,
    "score": 85,
    "matched_skills": ["Python", "AWS"],
    "why_match": "Strong Python background with 5 years AWS experience",
    "concerns": ["Location mismatch"],
    "reasoning": "Detailed explanation of fit"
  }}
]

SCORING CRITERIA:
- Skills match: 40 points
- Experience level: 30 points  
- Location match: 15 points
- Certifications: 15 points

Return JSON array only:"""

        try:
            result = self.llm.invoke(matching_prompt)
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            # Clean and parse
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*', '', result_text)
            json_match = re.search(r'\[[\s\S]*\]', result_text)
            
            if json_match:
                json_str = json_match.group(0)
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                analysis_list = json.loads(json_str)
                
                # Merge analysis with candidates
                for i, candidate in enumerate(candidates):
                    if i < len(analysis_list):
                        analysis = analysis_list[i]
                        candidate['match_score'] = analysis.get('score', 50)
                        candidate['matched_skills'] = analysis.get('matched_skills', [])
                        candidate['why_match'] = analysis.get('why_match', '')
                        candidate['concerns'] = analysis.get('concerns', [])
                        candidate['reasoning'] = analysis.get('reasoning', '')
                    else:
                        candidate['match_score'] = 50
                
                # Sort by score
                candidates.sort(key=lambda x: x.get('match_score', 0), reverse=True)
                
                state["analyzed_candidates"] = candidates
                
                log_msg = f"✓ Top score: {candidates[0].get('match_score', 0)}/100"
                print(f"   {log_msg}")
                state["execution_log"].append(log_msg)
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            state["analyzed_candidates"] = candidates
        
        state["next_agent"] = "END"
        return state


# ============== CREATE LANGGRAPH ==============
def create_arytic_graph():
    query_agent = QueryUnderstandingAgent(llm)
    search_agent = IntelligentSearchAgent(qdrant_client, embeddings)
    matching_agent = IntelligentMatchingAgent(llm)
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("query", query_agent)
    workflow.add_node("search", search_agent)
    workflow.add_node("analysis", matching_agent)
    
    workflow.set_entry_point("query")
    workflow.add_edge("query", "search")
    
    def route_from_search(state):
        return state.get("next_agent", "analysis")
    
    workflow.add_conditional_edges(
        "search",
        route_from_search,
        {
            "analysis": "analysis",
            END: END
        }
    )
    
    workflow.add_edge("analysis", END)
    
    return workflow.compile()

arytic_graph = create_arytic_graph()


# ============== DOCUMENT PROCESSING ==============

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "".join([page.extract_text() for page in pdf_reader.pages])
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error: {str(e)}"


async def process_resume_intelligent(file_content: bytes, filename: str, extension: str):
    """Process resume with intelligent LLM extraction"""
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{extension}")
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        loop = asyncio.get_event_loop()
        if extension == '.pdf':
            text = await loop.run_in_executor(executor, extract_text_from_pdf, file_path)
        elif extension == '.docx':
            text = await loop.run_in_executor(executor, extract_text_from_docx, file_path)
        else:
            return None
        
        if "Error" in text or len(text) < 50:
            print(f"❌ Failed: {filename}")
            return None
        
        print(f"📄 Processing: {filename}")
        
        # Use intelligent extraction agent
        extraction_agent = IntelligentExtractionAgent(llm)
        resume_data = await loop.run_in_executor(
            executor, 
            extraction_agent.extract_resume_data, 
            text, 
            filename
        )
        
        resume_info = {
            "id": file_id,
            "filename": filename,
            **resume_data,
            "uploaded_at": datetime.now().isoformat()
        }
        
        # Create rich embedding from extracted data
        embedding_text = f"""
        {resume_info['name']} {resume_info['current_role']} 
        Skills: {' '.join(resume_info['skills'][:20])}
        Certifications: {' '.join(resume_info['certifications'])}
        Location: {resume_info['location']}
        Experience: {resume_info['years_experience']} years
        Education: {resume_info['degree']} {resume_info['university']}
        Companies: {' '.join(resume_info['previous_companies'][:3])}
        """.strip()
        
        embedding = await loop.run_in_executor(executor, embeddings.embed_query, embedding_text)
        
        point = PointStruct(
            id=file_id,
            vector=embedding,
            payload=resume_info
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        
        return resume_info
        
    except Exception as e:
        print(f"❌ Error: {filename}: {e}")
        return None


# ============== API ENDPOINTS ==============

@app.get("/")
async def root():
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    return {
        "message": "Arytic - Intelligent Resume RAG",
        "version": "6.0",
        "features": [
            "✅ LLM-driven extraction - learns from resume structure",
            "✅ 3 specialized agents (Query, Search, Matching)",
            "✅ Semantic search with embeddings",
            "✅ Intelligent matching with reasoning",
            "✅ Context-aware ranking",
            "✅ No hardcoded patterns"
        ],
        "stats": {
            "total_resumes": collection_info.points_count,
            "status": "operational"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process hiring queries"""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    print(f"\n{'='*80}")
    print(f"🚀 QUERY: {request.message}")
    print(f"{'='*80}")
    
    try:
        initial_state = AgentState(
            messages=[HumanMessage(content=request.message)],
            user_query=request.message,
            parsed_requirements={},
            search_results=[],
            analyzed_candidates=[],
            final_recommendation="",
            next_agent="query",
            execution_log=[]
        )
        
        final_state = arytic_graph.invoke(initial_state)
        
        candidates = final_state.get("analyzed_candidates", [])
        requirements = final_state.get("parsed_requirements", {})
        
        if not candidates:
            response = "❌ No matching candidates found."
        else:
            response = generate_response(request.message, requirements, candidates)
        
        print(f"\n{'='*80}")
        print(f"✅ Found {len(candidates)} candidates")
        print(f"{'='*80}\n")
        
        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            execution_details={
                "requirements": requirements,
                "candidates_found": len(candidates),
                "execution_log": final_state.get("execution_log", [])
            }
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return ChatResponse(
            response=f"Error: {str(e)}",
            conversation_id=conversation_id,
            execution_details={"error": str(e)}
        )


def generate_response(query: str, requirements: dict, candidates: list) -> str:
    """Generate detailed response"""
    
    response = f"🎯 MATCHING RESULTS\n{'='*80}\n\n"
    response += f"📝 Query: \"{query}\"\n\n"
    
    if requirements:
        response += "🔍 Requirements:\n"
        for key, value in requirements.items():
            if value:
                response += f"   • {key}: {value}\n"
        response += "\n"
    
    response += f"👥 TOP {min(len(candidates), TOP_CANDIDATES)} CANDIDATES:\n\n"
    
    for i, c in enumerate(candidates[:TOP_CANDIDATES], 1):
        response += f"{'='*80}\n"
        response += f"#{i} - {c.get('name', 'N/A')}\n"
        response += f"{'='*80}\n"
        response += f"🎯 Match Score: {c.get('match_score', 0)}/100\n\n"
        
        response += "💼 Profile:\n"
        response += f"   • Role: {c.get('current_role', 'N/A')}\n"
        response += f"   • Experience: {c.get('years_experience', 0)} years\n"
        response += f"   • Location: {c.get('location', 'N/A')}\n"
        response += f"   • Education: {c.get('degree', 'N/A')}"
        if c.get('university') != 'N/A':
            response += f" - {c.get('university')}\n"
        else:
            response += "\n"
        
        if c.get('previous_companies'):
            response += f"   • Companies: {', '.join(c.get('previous_companies', [])[:3])}\n"
        response += "\n"
        
        if c.get('skills'):
            response += "🛠️ Skills:\n"
            skills = c.get('skills', [])[:15]
            response += f"   {', '.join(skills)}\n"
            if len(c.get('skills', [])) > 15:
                response += f"   ... and {len(c.get('skills', [])) - 15} more\n"
            response += "\n"
        
        if c.get('certifications'):
            response += "📜 Certifications:\n"
            response += f"   • {', '.join(c.get('certifications', []))}\n\n"
        
        if c.get('matched_skills'):
            response += "✅ Matched Skills:\n"
            response += f"   • {', '.join(c.get('matched_skills', []))}\n\n"
        
        if c.get('why_match'):
            response += "💡 Why Good Match:\n"
            response += f"   {c.get('why_match')}\n\n"
        
        if c.get('reasoning'):
            response += "🤖 Analysis:\n"
            response += f"   {c.get('reasoning')}\n\n"
        
        if c.get('concerns'):
            response += "⚠️ Considerations:\n"
            for concern in c.get('concerns', []):
                response += f"   • {concern}\n"
            response += "\n"
        
        response += "📧 Contact:\n"
        response += f"   • Email: {c.get('email', 'N/A')}\n"
        response += f"   • Phone: {c.get('phone', 'N/A')}\n\n"
    
    # Top recommendation
    if candidates:
        top = candidates[0]
        response += f"{'='*80}\n"
        response += "🏆 TOP RECOMMENDATION\n"
        response += f"{'='*80}\n\n"
        response += f"**{top.get('name')}** - Score: {top.get('match_score', 0)}/100\n\n"
        
        if top.get('why_match'):
            response += f"**Why?** {top.get('why_match')}\n\n"
        
        response += f"**Contact:** {top.get('email', 'N/A')}\n"
    
    return response


@app.post("/upload", response_model=UploadResponse)
async def upload_resumes(files: List[UploadFile] = File(...)):
    """Upload and process resumes"""
    print(f"\n📤 Uploading {len(files)} resumes...")
    
    file_data = []
    for file in files:
        content = await file.read()
        extension = os.path.splitext(file.filename)[1]
        file_data.append((content, file.filename, extension))
    
    tasks = []
    for content, filename, extension in file_data:
        task = process_resume_intelligent(content, filename, extension)
        tasks.append(task)
    
    processed_resumes = []
    for i in range(0, len(tasks), MAX_CONCURRENT_UPLOADS):
        batch = tasks[i:i+MAX_CONCURRENT_UPLOADS]
        print(f"📦 Batch {i//MAX_CONCURRENT_UPLOADS + 1} ({len(batch)} resumes)...")
        results = await asyncio.gather(*batch, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                print(f"⚠️ Error: {result}")
            elif result:
                processed_resumes.append(result)
    
    print(f"\n✅ Processed {len(processed_resumes)}/{len(files)} resumes\n")
    
    return UploadResponse(
        status="success",
        processed=len(processed_resumes),
        resumes=processed_resumes
    )

@app.get("/resumes")
async def get_resumes():
    """Get all resumes"""
    try:
        results = qdrant_client.scroll(collection_name=COLLECTION_NAME, limit=100)
        resumes = [point.payload for point in results[0]]
        return {"resumes": resumes, "count": len(resumes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """System statistics"""
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "total_resumes": collection_info.points_count,
            "architecture": "LLM-driven RAG (3 Agents)",
            "model": OLLAMA_MODEL,
            "status": "operational"
        }
    except Exception as e:
        return {"total_resumes": 0, "status": "error", "message": str(e)}

@app.delete("/resumes/clear")
async def clear_resumes():
    """Clear all resumes"""
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
        initialize_qdrant()
        return {"message": "All resumes cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "Arytic Intelligent RAG",
        "version": "6.0"
    }

@app.on_event("startup")
async def startup():
    print("\n" + "="*80)
    print("🚀 ARYTIC - INTELLIGENT RESUME RAG")
    print("="*80)
    
    print("\n⚡ KEY FEATURES:")
    print(f"   ✅ LLM-driven extraction - learns from resume structure")
    print(f"   ✅ No hardcoded patterns - agent understands context")
    print(f"   ✅ 3 Specialized Agents:")
    print(f"      1️⃣  Query Understanding - extracts requirements")
    print(f"      2️⃣  Intelligent Search - semantic matching")
    print(f"      3️⃣  Intelligent Matching - explains fit with reasoning")
    print(f"   ✅ Rich embeddings from all extracted fields")
    print(f"   ✅ Context-aware ranking and scoring")
    
    print(f"\n🔧 Configuration:")
    print(f"   • Model: {OLLAMA_MODEL}")
    print(f"   • Temperature: 0.1 (balanced accuracy)")
    print(f"   • CPU Threads: {CPU_THREADS}")
    print(f"   • Context: {CONTEXT_SIZE} tokens")
    print(f"   • Embedding Model: {EMBEDDING_MODEL}")
    
    print(f"\n🤖 Testing Ollama...")
    try:
        test_response = llm.invoke("OK")
        print(f"   ✅ Ollama responding")
    except Exception as e:
        print(f"   ⚠️ Ollama issue: {e}")
        print(f"   💡 Run: ollama serve")
    
    print(f"\n🧠 Testing Embeddings...")
    try:
        test_emb = embeddings.embed_query("test")
        print(f"   ✅ Embeddings working (dim: {len(test_emb)})")
    except Exception as e:
        print(f"   ⚠️ Embeddings issue: {e}")
    
    print("\n" + "="*80)
    print("🌐 Server: http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print("="*80 + "\n")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")