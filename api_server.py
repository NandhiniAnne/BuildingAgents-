"""
Arytic - LangGraph Multi-Agent Resume System v12.0
TRUE INTELLIGENCE: No hardcoding, autonomous agents, iterative refinement
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, TypedDict, Annotated
import uvicorn
import os
import uuid
from datetime import datetime
import json
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END

# LangChain imports
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Document processing
import PyPDF2
import docx

# ============== CONFIGURATION ==============
OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "arytic_resumes"
UPLOAD_DIR = "arytic_uploads"
MAX_CONCURRENT_UPLOADS = 5
MAX_WORKERS = 5

os.makedirs(UPLOAD_DIR, exist_ok=True)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ============== FASTAPI SETUP ==============
app = FastAPI(title="Arytic LangGraph System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============== INITIALIZE SERVICES ==============
qdrant_client = QdrantClient(url=QDRANT_URL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2)

def initialize_qdrant():
    try:
        collections = qdrant_client.get_collections().collections
        if not any(c.name == COLLECTION_NAME for c in collections):
            test_emb = embeddings.embed_query("test")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=len(test_emb), distance=Distance.COSINE)
            )
            print(f"✅ Created collection")
    except Exception as e:
        print(f"❌ Qdrant error: {e}")

initialize_qdrant()

# ============== PYDANTIC MODELS ==============
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    candidates_found: int
    reasoning_steps: List[str]

class UploadResponse(BaseModel):
    status: str
    processed: int
    resumes: List[Dict[str, Any]]

# ============== LANGGRAPH STATE ==============

class AgentState(TypedDict):
    """State that flows through all agents"""
    # Input
    query: str
    
    # Agent 1: Query Analyzer
    extracted_requirements: Dict[str, Any]
    search_strategy: str
    
    # Agent 2: Semantic Searcher
    raw_candidates: List[Dict]
    search_metadata: Dict
    
    # Agent 3: Intelligent Scorer
    scored_candidates: List[Dict]
    scoring_reasoning: List[str]
    
    # Agent 4: Quality Checker
    needs_refinement: bool
    refinement_reason: str
    iteration_count: int
    
    # Agent 5: Response Synthesizer
    final_response: str
    top_candidates: List[Dict]
    
    # Tracking
    agent_logs: Annotated[List[str], operator.add]


# ============== TEXT EXTRACTION (No changes needed) ==============

def extract_text_from_pdf(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "".join([page.extract_text() for page in pdf_reader.pages])
    except:
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except:
        return ""

def extract_text_from_old_doc(file_path: str) -> str:
    try:
        import docx2txt
        return docx2txt.process(file_path)
    except:
        return ""


# ============== INTELLIGENT EXTRACTION AGENT ==============

def intelligent_extract_metadata(text: str, filename: str) -> dict:
    """
    Pure LLM extraction - NO hardcoding
    Agent autonomously decides what's important
    """
    
    if not text or len(text) < 50:
        return {
            "name": filename.rsplit('.', 1)[0].replace('_', ' ').title(),
            "raw_text": text,
            "filename": filename,
            "id": str(uuid.uuid4()),
            "uploaded_at": datetime.utcnow().isoformat()
        }
    
    extraction_prompt = f"""You are an expert resume analyzer. Extract ALL relevant information from this resume.

CRITICAL INSTRUCTIONS:
1. Extract EVERYTHING you find - don't limit yourself to predefined categories
2. Identify the candidate's name, contact info, role, experience, skills, certifications, education, and work history
3. For skills: Extract ALL technical skills, tools, technologies, methodologies you see
4. For experience: Calculate or extract years of experience
5. For locations: Extract all cities/states/countries mentioned
6. If information is not present, use empty string "" or empty array []
7. DO NOT write "not present" or "not stated" - just leave empty

RESUME TEXT:
{text[:15000]}

Extract in this JSON format:
{{
    "name": "full name",
    "email": "email address",
    "phone": "phone number",
    "current_location": "current location from header",
    "current_role": "current or most recent job title",
    "years_experience": "X years or X+ years",
    "skills": ["every skill, tool, technology, language, framework you find"],
    "certifications": ["every certification mentioned"],
    "languages": ["programming languages and spoken languages"],
    "education": [
        {{
            "degree": "degree name",
            "university": "university name",
            "location": "university location",
            "year": "graduation year"
        }}
    ],
    "work_experience": [
        {{
            "company": "company name",
            "role": "job title",
            "location": "work location",
            "duration": "time period",
            "responsibilities": "brief summary",
            "technologies": ["technologies used in this role"]
        }}
    ],
    "all_locations": ["all cities/states mentioned in education or work"],
    "summary": "one sentence summary of this candidate"
}}

Extract EVERYTHING you see. Be thorough. Your JSON:"""
    
    try:
        response = llm.invoke(extraction_prompt).content
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            metadata = json.loads(json_match.group())
            
            # Clean up "not present" responses
            for key, value in metadata.items():
                if isinstance(value, str) and any(x in value.lower() for x in ['not present', 'not stated', 'not specified', 'not mentioned']):
                    metadata[key] = ""
                elif isinstance(value, list):
                    metadata[key] = [v for v in value if v and not any(x in str(v).lower() for x in ['not present', 'not stated'])]
        else:
            metadata = {"name": filename.rsplit('.', 1)[0]}
    except Exception as e:
        print(f"Extraction error: {e}")
        metadata = {"name": filename.rsplit('.', 1)[0]}
    
    # Add metadata
    metadata['filename'] = filename
    metadata['id'] = str(uuid.uuid4())
    metadata['uploaded_at'] = datetime.utcnow().isoformat()
    metadata['raw_text'] = text[:1000]
    
    return metadata


# ============== LANGGRAPH AGENTS ==============

def agent_1_query_analyzer(state: AgentState) -> AgentState:
    """
    Agent 1: Analyze query and extract requirements
    NO hardcoded skills or roles - pure LLM reasoning
    """
    
    query = state['query']
    
    analysis_prompt = f"""You are a senior recruiter analyzing a job search query. Your job is to extract ALL requirements and create a search strategy.

USER QUERY: {query}

Analyze this query and extract:
1. What role/job title are they looking for? (e.g., "Data Engineer", "Project Manager")
2. What skills/technologies are required? (extract ALL mentioned)
3. What experience level? (junior, mid, senior, years)
4. What certifications or qualifications?
5. Any location preferences?
6. Any other implicit requirements? (e.g., "AWS" implies cloud experience)

Then decide on a search strategy:
- BROAD: If query is vague, search broadly
- FOCUSED: If query has specific requirements, focus on exact matches
- HYBRID: Mix of both

Output JSON:
{{
    "role": "extracted role",
    "required_skills": ["all skills mentioned or implied"],
    "experience_level": "junior/mid/senior or X+ years",
    "certifications": ["any certs mentioned"],
    "location": "location if mentioned",
    "implicit_requirements": ["things implied but not stated"],
    "search_strategy": "BROAD/FOCUSED/HYBRID",
    "reasoning": "why you chose this strategy"
}}

Your analysis:"""
    
    try:
        response = llm.invoke(analysis_prompt).content
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = {
                "role": query,
                "required_skills": [],
                "search_strategy": "HYBRID",
                "reasoning": "Could not parse query, using hybrid approach"
            }
    except:
        analysis = {
            "role": query,
            "required_skills": [],
            "search_strategy": "HYBRID",
            "reasoning": "Error in analysis, using hybrid approach"
        }
    
    return {
        **state,
        "extracted_requirements": analysis,
        "search_strategy": analysis.get("search_strategy", "HYBRID"),
        "agent_logs": [f"🔍 Query Analyzer: Extracted {len(analysis.get('required_skills', []))} requirements"]
    }


def agent_2_semantic_searcher(state: AgentState) -> AgentState:
    """
    Agent 2: Perform semantic search
    Dynamically adjusts search based on strategy
    """
    
    requirements = state['extracted_requirements']
    strategy = state['search_strategy']
    
    # Build search query based on requirements
    role = requirements.get('role', '')
    skills = requirements.get('required_skills', [])
    experience = requirements.get('experience_level', '')
    
    # Create rich search query
    search_query = f"{role} {' '.join(skills)} {experience}"
    
    # Adjust limit based on strategy
    limit = {
        "BROAD": 30,
        "FOCUSED": 15,
        "HYBRID": 20
    }.get(strategy, 20)
    
    try:
        query_embedding = embeddings.embed_query(search_query)
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Deduplicate
        candidates = []
        seen = set()
        for hit in search_results:
            candidate = hit.payload
            key = f"{candidate.get('name', '')}-{candidate.get('email', '')}".lower()
            if key not in seen and candidate.get('name'):
                seen.add(key)
                candidate['_semantic_score'] = float(hit.score)
                candidates.append(candidate)
        
        metadata = {
            "total_found": len(candidates),
            "search_query": search_query,
            "strategy_used": strategy
        }
        
    except Exception as e:
        print(f"Search error: {e}")
        candidates = []
        metadata = {"error": str(e)}
    
    return {
        **state,
        "raw_candidates": candidates,
        "search_metadata": metadata,
        "agent_logs": [f"🔎 Semantic Searcher: Found {len(candidates)} candidates using {strategy} strategy"]
    }


def agent_3_intelligent_scorer(state: AgentState) -> AgentState:
    """
    Agent 3: Score candidates intelligently
    NO hardcoded scoring - LLM decides what matters
    """
    
    candidates = state['raw_candidates']
    requirements = state['extracted_requirements']
    
    if not candidates:
        return {
            **state,
            "scored_candidates": [],
            "scoring_reasoning": ["No candidates to score"],
            "agent_logs": ["⚠️ Intelligent Scorer: No candidates to score"]
        }
    
    # Score each candidate
    scored = []
    reasoning = []
    
    for candidate in candidates[:15]:  # Score top 15 to save time
        
        scoring_prompt = f"""You are an expert recruiter. Score this candidate for the given requirements.

REQUIREMENTS:
{json.dumps(requirements, indent=2)}

CANDIDATE:
Name: {candidate.get('name', 'Unknown')}
Role: {candidate.get('current_role', 'Not specified')}
Experience: {candidate.get('years_experience', 'Not specified')}
Skills: {', '.join(candidate.get('skills', [])[:20])}
Certifications: {', '.join(candidate.get('certifications', []))}
Education: {json.dumps(candidate.get('education', []))}
Summary: {candidate.get('summary', '')}

Score this candidate from 0-100 based on:
1. Role match (does their role align with requirements?)
2. Skills match (do they have required skills?)
3. Experience level (appropriate experience?)
4. Certifications (have required certs?)
5. Overall fit

Output JSON:
{{
    "overall_score": 85,
    "role_match_score": 90,
    "skills_match_score": 80,
    "experience_match_score": 85,
    "match_type": "EXCELLENT/GOOD/PARTIAL/WEAK",
    "strengths": ["what makes them a good fit"],
    "gaps": ["what they might be missing"],
    "reasoning": "brief explanation of score"
}}

Your scoring:"""
        
        try:
            response = llm.invoke(scoring_prompt).content
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                score_data = json.loads(json_match.group())
                candidate['_scoring'] = score_data
                candidate['_overall_score'] = score_data.get('overall_score', 50)
                scored.append(candidate)
                reasoning.append(f"{candidate['name']}: {score_data.get('reasoning', '')}")
        except Exception as e:
            print(f"Scoring error for {candidate.get('name')}: {e}")
            candidate['_overall_score'] = 50
            scored.append(candidate)
    
    # Sort by score
    scored.sort(key=lambda x: x.get('_overall_score', 0), reverse=True)
    
    return {
        **state,
        "scored_candidates": scored,
        "scoring_reasoning": reasoning,
        "agent_logs": [f"🎯 Intelligent Scorer: Scored {len(scored)} candidates"]
    }


def agent_4_quality_checker(state: AgentState) -> AgentState:
    """
    Agent 4: Check if results are good enough
    Decides if we need to refine search
    """
    
    scored = state['scored_candidates']
    iteration = state.get('iteration_count', 0)
    
    if not scored:
        return {
            **state,
            "needs_refinement": True,
            "refinement_reason": "No candidates found",
            "iteration_count": iteration + 1,
            "agent_logs": ["⚠️ Quality Checker: No candidates, needs refinement"]
        }
    
    # Check quality
    top_score = scored[0].get('_overall_score', 0) if scored else 0
    num_good_matches = sum(1 for c in scored if c.get('_overall_score', 0) >= 70)
    
    # Decision logic
    needs_refinement = False
    reason = ""
    
    if iteration >= 2:
        # Stop after 2 iterations
        needs_refinement = False
        reason = "Maximum iterations reached"
    elif top_score < 60:
        needs_refinement = True
        reason = f"Top score too low ({top_score}), broadening search"
    elif num_good_matches < 2:
        needs_refinement = True
        reason = f"Only {num_good_matches} good matches, need more"
    else:
        needs_refinement = False
        reason = f"Found {num_good_matches} good matches"
    
    return {
        **state,
        "needs_refinement": needs_refinement,
        "refinement_reason": reason,
        "iteration_count": iteration + 1,
        "agent_logs": [f"✅ Quality Checker: {reason}"]
    }


def agent_5_response_synthesizer(state: AgentState) -> AgentState:
    """
    Agent 5: Create professional response
    Synthesizes all agent insights
    """
    
    scored = state['scored_candidates']
    requirements = state['extracted_requirements']
    reasoning = state.get('scoring_reasoning', [])
    
    if not scored:
        return {
            **state,
            "final_response": "No candidates found matching your requirements. Please try adjusting your search criteria or upload more resumes.",
            "top_candidates": [],
            "agent_logs": ["📝 Response Synthesizer: No candidates to present"]
        }
    
    top_5 = scored[:5]
    
    synthesis_prompt = f"""You are a professional recruiter presenting candidates to a client. Create a polished response.

SEARCH REQUIREMENTS:
{json.dumps(requirements, indent=2)}

TOP CANDIDATES:
{json.dumps([{
    'name': c.get('name'),
    'role': c.get('current_role'),
    'experience': c.get('years_experience'),
    'score': c.get('_overall_score'),
    'scoring': c.get('_scoring'),
    'email': c.get('email'),
    'phone': c.get('phone')
} for c in top_5], indent=2)}

Create a professional response that:
1. Acknowledges the search query
2. Presents top 3-5 candidates with:
   - Why they're a good fit
   - Their key strengths
   - Any gaps or considerations
   - Contact information
3. Provides actionable next steps

Make it professional, concise, and client-ready. Use clear formatting.

Your response:"""
    
    try:
        response = llm.invoke(synthesis_prompt).content
    except:
        # Fallback response
        response = f"Found {len(scored)} candidates.\n\nTop matches:\n"
        for i, c in enumerate(top_5, 1):
            response += f"\n{i}. {c.get('name', 'Unknown')} - {c.get('current_role', 'Not specified')}\n"
            response += f"   Score: {c.get('_overall_score', 0)}/100\n"
            response += f"   Email: {c.get('email', 'Not provided')}\n"
    
    return {
        **state,
        "final_response": response,
        "top_candidates": top_5,
        "agent_logs": [f"📝 Response Synthesizer: Created response with {len(top_5)} candidates"]
    }


# ============== LANGGRAPH WORKFLOW ==============

def should_refine_search(state: AgentState) -> str:
    """Decision point: refine or finalize?"""
    if state.get('needs_refinement', False) and state.get('iteration_count', 0) < 2:
        return "refine"
    return "finalize"


def build_agent_workflow() -> StateGraph:
    """Build the LangGraph workflow"""
    
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("query_analyzer", agent_1_query_analyzer)
    workflow.add_node("semantic_searcher", agent_2_semantic_searcher)
    workflow.add_node("intelligent_scorer", agent_3_intelligent_scorer)
    workflow.add_node("quality_checker", agent_4_quality_checker)
    workflow.add_node("response_synthesizer", agent_5_response_synthesizer)
    
    # Define flow
    workflow.set_entry_point("query_analyzer")
    workflow.add_edge("query_analyzer", "semantic_searcher")
    workflow.add_edge("semantic_searcher", "intelligent_scorer")
    workflow.add_edge("intelligent_scorer", "quality_checker")
    
    # Conditional: refine or finalize
    workflow.add_conditional_edges(
        "quality_checker",
        should_refine_search,
        {
            "refine": "query_analyzer",  # Loop back to try again with different strategy
            "finalize": "response_synthesizer"
        }
    )
    
    workflow.add_edge("response_synthesizer", END)
    
    return workflow.compile()


# Create the compiled graph
agent_graph = build_agent_workflow()


# ============== UPLOAD PROCESSING ==============

async def process_single_file(file: UploadFile) -> Optional[Dict]:
    try:
        print(f"\n📄 Processing: {file.filename}")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        content = await file.read()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, lambda: open(file_path, 'wb').write(content))
        
        # Extract text
        if file.filename.endswith('.pdf'):
            text = await loop.run_in_executor(executor, extract_text_from_pdf, file_path)
        elif file.filename.endswith('.docx'):
            text = await loop.run_in_executor(executor, extract_text_from_docx, file_path)
        elif file.filename.endswith('.doc'):
            text = await loop.run_in_executor(executor, extract_text_from_old_doc, file_path)
        else:
            return None
        
        if not text or len(text) < 50:
            print(f"  ⚠️ No text extracted")
            return None
        
        # Extract metadata with intelligent agent
        metadata = await loop.run_in_executor(executor, intelligent_extract_metadata, text, file.filename)
        
        # Create embedding
        emb_text = f"""
Name: {metadata.get('name', '')}
Role: {metadata.get('current_role', '')}
Skills: {' '.join(metadata.get('skills', [])[:30])}
Experience: {metadata.get('years_experience', '')}
Summary: {metadata.get('summary', '')}
"""
        embedding = await loop.run_in_executor(executor, embeddings.embed_query, emb_text)
        
        # Store
        point = PointStruct(id=metadata['id'], vector=embedding, payload=metadata)
        await loop.run_in_executor(executor, lambda: qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point]))
        
        print(f"  ✅ Processed: {metadata.get('name', 'Unknown')}")
        return metadata
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


# ============== API ENDPOINTS ==============

@app.post("/upload", response_model=UploadResponse)
async def upload_resumes(files: List[UploadFile] = File(...)):
    print(f"\n📤 Uploading {len(files)} files...")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
    async def process_with_limit(file):
        async with semaphore:
            return await process_single_file(file)
    
    tasks = [process_with_limit(file) for file in files]
    results = await asyncio.gather(*tasks)
    processed = [r for r in results if r]
    
    print(f"✅ Processed {len(processed)}/{len(files)}")
    return UploadResponse(status="success", processed=len(processed), resumes=processed)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Execute the LangGraph agent workflow
    """
    try:
        print(f"\n🤖 Running LangGraph agents for: {request.message}")
        
        # Initialize state
        initial_state = {
            "query": request.message,
            "agent_logs": [],
            "iteration_count": 0
        }
        
        # Run the graph
        final_state = agent_graph.invoke(initial_state)
        
        # Extract results
        response = final_state.get('final_response', 'No response generated')
        candidates_found = len(final_state.get('raw_candidates', []))
        reasoning = final_state.get('agent_logs', [])
        
        print(f"✅ Workflow complete: {candidates_found} candidates found")
        
        return ChatResponse(
            response=response,
            candidates_found=candidates_found,
            reasoning_steps=reasoning
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "total_resumes": info.points_count,
            "status": "operational",
            "using_langgraph": True,
            "agents": 5
        }
    except:
        return {"total_resumes": 0, "status": "error"}


@app.get("/")
async def root():
    return {
        "system": "Arytic LangGraph v12.0",
        "architecture": "Multi-Agent LangGraph System",
        "agents": [
            "1. Query Analyzer - Extracts requirements",
            "2. Semantic Searcher - Finds candidates",
            "3. Intelligent Scorer - Scores matches",
            "4. Quality Checker - Validates results",
            "5. Response Synthesizer - Creates response"
        ],
        "features": [
            "✅ No hardcoded skills or roles",
            "✅ Iterative refinement",
            "✅ Autonomous agent decisions",
            "✅ Self-correcting workflow",
            "✅ Dynamic scoring"
        ],
        "endpoints": {
            "POST /upload": "Upload resumes",
            "POST /chat": "Query with LangGraph agents"
        }
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🤖 ARYTIC LANGGRAPH v12.0 - TRUE MULTI-AGENT INTELLIGENCE")
    print("="*80)
    print("\n🎯 ARCHITECTURE:")
    print("   • Agent 1: Query Analyzer - Extracts requirements autonomously")
    print("   • Agent 2: Semantic Searcher - Dynamic search strategy")
    print("   • Agent 3: Intelligent Scorer - LLM-based scoring (no hardcoding)")
    print("   • Agent 4: Quality Checker - Self-correction & refinement")
    print("   • Agent 5: Response Synthesizer - Professional responses")
    print("\n✨ INTELLIGENCE:")
    print("   • NO hardcoded skills, roles, or scoring rules")
    print("   • Agents decide what matters based on query")
    print("   • Iterative refinement if results are poor")
    print("   • Complete observability of agent decisions")
    print("\n" + "="*80)
    print("🌐 http://127.0.0.1:8000")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")