"""
LangGraph Multi-Agent Resume Intelligence System - PRODUCTION VERSION
Intelligent Query Understanding with Dynamic Routing
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
import asyncio
import operator
import re

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
OLLAMA_MODEL = "mistral:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "resumes_collection"
UPLOAD_DIR = "uploaded_resumes"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============== FASTAPI SETUP ==============
app = FastAPI(title="LangGraph Multi-Agent Resume Intelligence - PRODUCTION")

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
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)

# Initialize Qdrant collection
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
            print(f"✅ Created collection (vector size: {VECTOR_SIZE})")
        else:
            print(f"✅ Using existing collection")
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
    agent_flow: List[str]
    execution_details: Dict[str, Any]

class UploadResponse(BaseModel):
    status: str
    processed: int
    resumes: List[Dict[str, Any]]

# ============== LANGGRAPH STATE ==============
class AgentState(TypedDict):
    """Shared state between all agents"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    query_intent: str  # NEW: What user wants to do
    target_names: List[str]  # NEW: Specific names mentioned
    search_results: List[Dict]
    analysis_results: Dict[str, Any]
    comparison_results: Dict[str, Any]
    final_recommendation: str
    next_agent: str
    iteration_count: int
    agent_decisions: List[str]
    execution_log: List[str]

# ============== QUERY UNDERSTANDING AGENT ==============

class QueryUnderstandingAgent:
    """NEW: Intelligent query parser that understands user intent"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "🧩 Query Understanding Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        """Analyze query and determine intent"""
        query = state["user_query"]
        
        log_msg = f"{self.name}: Analyzing query intent"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        understanding_prompt = f"""You are an intelligent query analyzer for a resume database system.

User Query: "{query}"

Analyze this query and determine:

1. INTENT: What does the user want?
   - SEARCH: Find candidates by skills/role/experience (e.g., "find Python developers", "who knows AWS")
   - GET_DETAILS: Get specific candidate information (e.g., "tell me about John", "Dexter's details", "what are Sarah's skills")
   - COMPARE: Compare multiple candidates (e.g., "compare John and Sarah", "who is better")
   - LIST: List all candidates or by category (e.g., "show all resumes", "list everyone")
   - STATS: Database statistics (e.g., "how many resumes", "show stats")
   - COUNT: Count specific types (e.g., "how many Python developers")

2. NAMES: Extract any person names mentioned (first name, last name, or full name)

3. KEYWORDS: Key search terms or skills mentioned

Respond in this EXACT format:
INTENT: [one of: SEARCH, GET_DETAILS, COMPARE, LIST, STATS, COUNT]
NAMES: [comma-separated list of names, or NONE]
KEYWORDS: [comma-separated keywords, or NONE]
CONFIDENCE: [HIGH/MEDIUM/LOW]

Examples:
Query: "find Python developers"
INTENT: SEARCH
NAMES: NONE
KEYWORDS: Python, developers
CONFIDENCE: HIGH

Query: "tell me about Dexter"
INTENT: GET_DETAILS
NAMES: Dexter
KEYWORDS: details, information
CONFIDENCE: HIGH

Query: "what are dexter details?"
INTENT: GET_DETAILS
NAMES: Dexter
KEYWORDS: details
CONFIDENCE: HIGH

Query: "compare John Smith and Sarah Johnson"
INTENT: COMPARE
NAMES: John Smith, Sarah Johnson
KEYWORDS: compare
CONFIDENCE: HIGH

Now analyze: "{query}"
"""
        
        result = self.llm.invoke(understanding_prompt)
        result_text = result.content if hasattr(result, 'content') else str(result)
        
        # Parse intent
        intent = "SEARCH"  # default
        names = []
        
        if "INTENT:" in result_text:
            intent_match = re.search(r'INTENT:\s*(\w+)', result_text)
            if intent_match:
                intent = intent_match.group(1).upper()
        
        if "NAMES:" in result_text:
            names_match = re.search(r'NAMES:\s*(.+)', result_text)
            if names_match:
                names_str = names_match.group(1).strip()
                if names_str != "NONE":
                    names = [n.strip() for n in names_str.split(',')]
        
        state["query_intent"] = intent
        state["target_names"] = names
        state["agent_decisions"].append(f"{self.name}: Intent={intent}, Names={names}")
        
        print(f"   Intent: {intent}")
        print(f"   Names: {names}")
        
        # Route based on intent
        if intent == "GET_DETAILS":
            state["next_agent"] = "details"
        elif intent == "COMPARE":
            state["next_agent"] = "compare_direct"
        elif intent == "LIST":
            state["next_agent"] = "list"
        elif intent == "STATS":
            state["next_agent"] = "stats"
        elif intent == "COUNT":
            state["next_agent"] = "search"
        else:  # SEARCH
            state["next_agent"] = "search"
        
        return state


# ============== DETAILS RETRIEVAL AGENT ==============

class DetailsAgent:
    """NEW: Retrieves specific candidate details by name"""
    
    def __init__(self, qdrant_client):
        self.qdrant = qdrant_client
        self.name = "📋 Details Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        """Get detailed information about specific candidate(s)"""
        names = state["target_names"]
        
        log_msg = f"{self.name}: Retrieving details for: {names}"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        if not names:
            state["final_recommendation"] = "Please specify which candidate you want details about."
            state["next_agent"] = "END"
            return state
        
        # Get all resumes
        results = self.qdrant.scroll(collection_name=COLLECTION_NAME, limit=100)
        
        found_candidates = []
        for point in results[0]:
            resume = point.payload
            resume_name = resume.get('name', '').lower()
            
            # Check if any target name matches
            for target_name in names:
                target_lower = target_name.lower()
                # Match first name, last name, or full name
                if (target_lower in resume_name or 
                    resume_name in target_lower or
                    any(part in resume_name for part in target_lower.split())):
                    found_candidates.append(resume)
                    break
        
        if not found_candidates:
            state["final_recommendation"] = f"I couldn't find any candidates matching: {', '.join(names)}. Please check the name and try again."
            state["next_agent"] = "END"
            return state
        
        # Format detailed response
        response = ""
        for resume in found_candidates:
            response += f"""
{'='*70}
👤 {resume.get('name', 'N/A')}
{'='*70}

📧 CONTACT INFORMATION:
   Email: {resume.get('email', 'N/A')}
   Phone: {resume.get('phone', 'N/A')}
   Location: {resume.get('location', 'N/A')}

💼 PROFESSIONAL SUMMARY:
   Current Role: {resume.get('current_role', 'N/A')}
   Years of Experience: {resume.get('years_experience', 0)} years
   Salary Expectations: {resume.get('salary_expectations', 'Not specified')}

🛠️ TECHNICAL SKILLS:
   {', '.join(resume.get('skills', [])) if resume.get('skills') else 'No skills listed'}

🏆 CERTIFICATIONS:
   {', '.join(resume.get('certifications', [])) if resume.get('certifications') else 'No certifications listed'}

🏢 PREVIOUS COMPANIES:
   {', '.join(resume.get('previous_companies', [])) if resume.get('previous_companies') else 'No previous companies listed'}

🎓 EDUCATION:
   Degree: {resume.get('degree', 'N/A')}
   University: {resume.get('university', 'N/A')}
   Graduation: {resume.get('graduation_year', 'N/A')}

📅 Uploaded: {resume.get('uploaded_at', 'N/A')}

"""
        
        state["final_recommendation"] = response
        state["next_agent"] = "END"
        return state


# ============== LIST AGENT ==============

class ListAgent:
    """NEW: Lists all candidates or filtered list"""
    
    def __init__(self, qdrant_client):
        self.qdrant = qdrant_client
        self.name = "📝 List Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        """List all candidates with summary"""
        log_msg = f"{self.name}: Listing all candidates"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        results = self.qdrant.scroll(collection_name=COLLECTION_NAME, limit=100)
        
        if not results[0]:
            state["final_recommendation"] = "No resumes in database yet. Please upload some resumes first."
            state["next_agent"] = "END"
            return state
        
        response = f"📋 ALL CANDIDATES ({len(results[0])} total):\n\n"
        
        for i, point in enumerate(results[0], 1):
            resume = point.payload
            response += f"{i}. **{resume.get('name', 'N/A')}**\n"
            response += f"   Role: {resume.get('current_role', 'N/A')}\n"
            response += f"   Experience: {resume.get('years_experience', 0)} years\n"
            response += f"   Location: {resume.get('location', 'N/A')}\n"
            response += f"   Skills: {', '.join(resume.get('skills', [])[:5])}\n"
            response += f"   Email: {resume.get('email', 'N/A')}\n\n"
        
        state["final_recommendation"] = response
        state["next_agent"] = "END"
        return state


# ============== STATS AGENT ==============

class StatsAgent:
    """NEW: Provides database statistics"""
    
    def __init__(self, qdrant_client):
        self.qdrant = qdrant_client
        self.name = "📊 Stats Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        """Generate comprehensive statistics"""
        log_msg = f"{self.name}: Generating statistics"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        collection_info = self.qdrant.get_collection(COLLECTION_NAME)
        results = self.qdrant.scroll(collection_name=COLLECTION_NAME, limit=100)
        
        if not results[0]:
            state["final_recommendation"] = "No statistics available. Database is empty."
            state["next_agent"] = "END"
            return state
        
        # Calculate stats
        total = len(results[0])
        total_exp = 0
        locations = {}
        skills_count = {}
        roles = {}
        
        for point in results[0]:
            resume = point.payload
            total_exp += resume.get('years_experience', 0)
            
            # Count locations
            loc = resume.get('location', 'Unknown')
            locations[loc] = locations.get(loc, 0) + 1
            
            # Count skills
            for skill in resume.get('skills', []):
                skills_count[skill] = skills_count.get(skill, 0) + 1
            
            # Count roles
            role = resume.get('current_role', 'Unknown')
            roles[role] = roles.get(role, 0) + 1
        
        avg_exp = total_exp / total if total > 0 else 0
        top_skills = sorted(skills_count.items(), key=lambda x: x[1], reverse=True)[:10]
        top_locations = sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5]
        top_roles = sorted(roles.items(), key=lambda x: x[1], reverse=True)[:5]
        
        response = f"""
📊 DATABASE STATISTICS
{'='*70}

📈 OVERVIEW:
   Total Resumes: {total}
   Average Experience: {avg_exp:.1f} years
   Total Experience: {total_exp} years combined

📍 TOP LOCATIONS:
"""
        for loc, count in top_locations:
            response += f"   • {loc}: {count} candidates\n"
        
        response += "\n🛠️ TOP SKILLS:\n"
        for skill, count in top_skills:
            response += f"   • {skill}: {count} candidates\n"
        
        response += "\n💼 TOP ROLES:\n"
        for role, count in top_roles:
            response += f"   • {role}: {count} candidates\n"
        
        state["final_recommendation"] = response
        state["next_agent"] = "END"
        return state


# ============== EXISTING AGENTS (Updated) ==============

class SearchAgent:
    """Specialized in semantic search and candidate discovery"""
    
    def __init__(self, llm, qdrant_client, embeddings):
        self.llm = llm
        self.qdrant = qdrant_client
        self.embeddings = embeddings
        self.name = "🔍 Search Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        """Execute search with intelligent decision making"""
        query = state["user_query"]
        iteration = state.get("iteration_count", 0)
        
        log_msg = f"{self.name}: Processing query (iteration {iteration})"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        # Generate embedding and search
        query_embedding = self.embeddings.embed_query(query)
        results = self.qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=20
        )
        
        if not results:
            state["next_agent"] = "END"
            state["final_recommendation"] = "No candidates found matching your search criteria."
            return state
        
        # Extract candidates
        candidates = []
        for r in results[:10]:
            candidate = r.payload.copy()
            candidate['relevance_score'] = round(r.score * 100, 1)
            candidates.append(candidate)
        
        state["search_results"] = candidates
        
        # Check if COUNT intent
        if state.get("query_intent") == "COUNT":
            state["final_recommendation"] = f"Found {len(results)} candidates matching '{query}'"
            state["next_agent"] = "END"
            return state
        
        # Otherwise continue to analysis
        state["next_agent"] = "analysis"
        return state


class AnalysisAgent:
    """Deep analysis with reasoning"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "🧠 Analysis Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        """Analyze candidates and make intelligent decisions"""
        candidates = state["search_results"][:5]
        query = state["user_query"]
        
        log_msg = f"{self.name}: Analyzing {len(candidates)} candidates"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        candidates_text = "\n\n".join([
            f"Candidate {i+1}:\n"
            f"Name: {c.get('name')}\n"
            f"Role: {c.get('current_role')}\n"
            f"Experience: {c.get('years_experience')}y\n"
            f"Skills: {', '.join(c.get('skills', [])[:8])}\n"
            f"Location: {c.get('location')}\n"
            f"Relevance: {c.get('relevance_score')}%"
            for i, c in enumerate(candidates)
        ])
        
        analysis_prompt = f"""You are an expert technical recruiter.

User Query: "{query}"

Candidates:
{candidates_text}

Provide detailed analysis:

1. QUALITY ASSESSMENT: Rate overall candidate quality (1-10)
2. TOP 3 PICKS: Rank top 3 candidates with specific reasons
3. KEY INSIGHTS: What makes each candidate strong/weak
4. DECISION: Do we need deeper comparison? (YES if top candidates are close, NO if clear winner)

Format:
QUALITY: [score]/10
TOP PICKS:
1. [Name] - [reason]
2. [Name] - [reason]  
3. [Name] - [reason]
INSIGHTS: [analysis]
DECISION: [YES/NO]
"""
        
        analysis = self.llm.invoke(analysis_prompt)
        analysis_text = analysis.content if hasattr(analysis, 'content') else str(analysis)
        
        state["analysis_results"] = {
            "full_analysis": analysis_text,
            "top_candidates": candidates[:3]
        }
        
        # Decision routing
        needs_comparison = "DECISION: YES" in analysis_text.upper()
        state["next_agent"] = "comparison" if needs_comparison else "ranking"
        
        return state


class ComparisonAgent:
    """Side-by-side detailed comparison"""
    
    def __init__(self, llm, qdrant_client):
        self.llm = llm
        self.qdrant = qdrant_client
        self.name = "⚖️ Comparison Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        """Deep comparative analysis"""
        
        # Check if direct comparison (from query understanding)
        if state.get("target_names") and len(state["target_names"]) >= 2:
            return self._compare_by_names(state)
        
        # Otherwise compare top candidates from search
        candidates = state["analysis_results"]["top_candidates"]
        query = state["user_query"]
        
        log_msg = f"{self.name}: Comparing top {len(candidates)} candidates"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        comparison_prompt = f"""You are an expert at comparing candidates.

User needs: "{query}"

Compare these candidates in detail:

{self._format_candidates(candidates)}

Provide:
1. SKILL COMPARISON: Side-by-side skill analysis
2. EXPERIENCE: Who has more relevant experience and why
3. STRENGTHS: Unique strengths of each candidate
4. RECOMMENDATION: Preliminary ranking with justification

Be specific and data-driven.
"""
        
        comparison = self.llm.invoke(comparison_prompt)
        comparison_text = comparison.content if hasattr(comparison, 'content') else str(comparison)
        
        state["comparison_results"] = {
            "full_comparison": comparison_text,
            "compared_candidates": candidates
        }
        
        state["next_agent"] = "ranking"
        return state
    
    def _compare_by_names(self, state: AgentState) -> AgentState:
        """Compare specific candidates by name"""
        names = state["target_names"]
        
        # Find candidates
        results = self.qdrant.scroll(collection_name=COLLECTION_NAME, limit=100)
        found = []
        
        for point in results[0]:
            resume = point.payload
            resume_name = resume.get('name', '').lower()
            for target in names:
                if target.lower() in resume_name:
                    found.append(resume)
                    break
        
        if len(found) < 2:
            state["final_recommendation"] = f"Need at least 2 candidates to compare. Found only: {[r['name'] for r in found]}"
            state["next_agent"] = "END"
            return state
        
        # Format comparison
        comparison_text = "DETAILED COMPARISON:\n\n"
        for i, candidate in enumerate(found, 1):
            comparison_text += f"**Candidate {i}: {candidate['name']}**\n"
            comparison_text += f"Role: {candidate.get('current_role')}\n"
            comparison_text += f"Experience: {candidate.get('years_experience')}y\n"
            comparison_text += f"Skills: {', '.join(candidate.get('skills', []))}\n"
            comparison_text += f"Location: {candidate.get('location')}\n\n"
        
        state["final_recommendation"] = comparison_text
        state["next_agent"] = "END"
        return state
    
    def _format_candidates(self, candidates):
        return "\n\n".join([
            f"**{c.get('name')}**\n"
            f"Role: {c.get('current_role')}\n"
            f"Experience: {c.get('years_experience')}y\n"
            f"Skills: {', '.join(c.get('skills', []))}\n"
            f"Location: {c.get('location')}"
            for c in candidates
        ])


class RankingAgent:
    """Final weighted scoring and recommendation"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "🏆 Ranking Agent"
    
    def __call__(self, state: AgentState) -> AgentState:
        """Generate final weighted ranking"""
        analysis = state.get("analysis_results", {}).get("full_analysis", "")
        comparison = state.get("comparison_results", {}).get("full_comparison", "")
        candidates = state["analysis_results"]["top_candidates"]
        query = state["user_query"]
        
        log_msg = f"{self.name}: Generating final rankings"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        ranking_prompt = f"""You are the final decision maker.

User Query: "{query}"

Analysis Summary:
{analysis[:500]}

{f"Comparison: {comparison[:500]}" if comparison else ""}

Candidates:
{self._format_brief(candidates)}

Apply weighted scoring:
- Technical Skills Match: 35%
- Years of Experience: 25%
- Company Background: 20%
- Education Quality: 15%
- Location Fit: 5%

Provide FINAL RECOMMENDATION:

**TOP CHOICE:** [Name]
**Score:** [X]/100
**Why:** [Clear reasoning with data points]

**Runner-Up:** [Name]
**Score:** [X]/100

**Key Differentiators:** [What makes #1 better than #2]

Be decisive and specific.
"""
        
        ranking = self.llm.invoke(ranking_prompt)
        ranking_text = ranking.content if hasattr(ranking, 'content') else str(ranking)
        
        state["final_recommendation"] = ranking_text
        state["next_agent"] = "END"
        return state
    
    def _format_brief(self, candidates):
        return "\n".join([
            f"{i+1}. {c.get('name')} - {c.get('current_role')} "
            f"({c.get('years_experience')}y, {c.get('location')})"
            for i, c in enumerate(candidates)
        ])


# ============== LANGGRAPH CONSTRUCTION ==============

def create_resume_graph():
    """Build the intelligent multi-agent graph"""
    
    # Create agents
    query_understanding = QueryUnderstandingAgent(llm)
    details = DetailsAgent(qdrant_client)
    list_agent = ListAgent(qdrant_client)
    stats = StatsAgent(qdrant_client)
    search = SearchAgent(llm, qdrant_client, embeddings)
    analysis = AnalysisAgent(llm)
    comparison = ComparisonAgent(llm, qdrant_client)
    ranking = RankingAgent(llm)
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_understanding", query_understanding)
    workflow.add_node("details", details)
    workflow.add_node("list", list_agent)
    workflow.add_node("stats", stats)
    workflow.add_node("search", search)
    workflow.add_node("analysis", analysis)
    workflow.add_node("comparison", comparison)
    workflow.add_node("ranking", ranking)
    
    # Set entry point
    workflow.set_entry_point("query_understanding")
    
    # Routing from query understanding
    def route_from_understanding(state):
        next_agent = state.get("next_agent", "search")
        if next_agent == "END":
            return END
        return next_agent
    
    workflow.add_conditional_edges(
        "query_understanding",
        route_from_understanding,
        {
            "details": "details",
            "list": "list",
            "stats": "stats",
            "search": "search",
            "compare_direct": "comparison",
            END: END
        }
    )
    
    # Direct termination nodes
    workflow.add_edge("details", END)
    workflow.add_edge("list", END)
    workflow.add_edge("stats", END)
    
    # Search flow
    def route_from_search(state):
        next_agent = state.get("next_agent", "analysis")
        if next_agent == "END":
            return END
        return next_agent
    
    workflow.add_conditional_edges(
        "search",
        route_from_search,
        {
            "analysis": "analysis",
            END: END
        }
    )
    
    # Analysis flow
    workflow.add_conditional_edges(
        "analysis",
        lambda state: state.get("next_agent", "ranking"),
        {
            "comparison": "comparison",
            "ranking": "ranking"
        }
    )
    
    # Comparison and ranking
    workflow.add_edge("comparison", "ranking")
    workflow.add_edge("ranking", END)
    
    return workflow.compile()

# Global graph instance
resume_graph = create_resume_graph()

# ============== RESUME PROCESSING ==============

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

def clean_json_response(text: str) -> str:
    """Clean LLM response to extract valid JSON"""
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```json?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        json_str = json_match.group(0)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = json_str.replace('\n', ' ')
        return json_str
    return text

async def process_resume_simple(file_content: bytes, filename: str, extension: str):
    """Process resume with improved extraction"""
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{extension}")
        
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        if extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif extension == '.docx':
            text = extract_text_from_docx(file_path)
        else:
            return None
        
        if "Error" in text or len(text) < 50:
            print(f"❌ Text extraction failed for {filename}")
            return None
        
        extract_prompt = f"""Extract resume information from this text and return ONLY valid JSON.

Text:
{text[:2500]}

Return this exact format with proper JSON syntax:
{{
  "name": "Full Name Here",
  "email": "email@example.com",
  "phone": "123-456-7890",
  "location": "City, State",
  "current_role": "Job Title",
  "years_experience": 5,
  "previous_companies": ["Company1", "Company2"],
  "skills": ["Skill1", "Skill2", "Skill3"],
  "certifications": ["Cert1"],
  "degree": "Degree Name",
  "university": "University Name",
  "graduation_year": "2020"
}}

IMPORTANT: Return ONLY valid JSON with proper quotes and no trailing commas."""
        
        result = llm.invoke(extract_prompt)
        result_text = result.content if hasattr(result, 'content') else str(result)
        
        cleaned_json = clean_json_response(result_text)
        
        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error for {filename}: {e}")
            data = {
                "name": filename.replace('.pdf', '').replace('.docx', ''),
                "email": "N/A",
                "current_role": "See resume",
                "years_experience": 0,
                "skills": ["See resume"]
            }
        
        resume_info = {
            "id": file_id,
            "filename": filename,
            "name": str(data.get("name", filename.replace('.pdf', '').replace('.docx', ''))),
            "email": str(data.get("email", "N/A")),
            "phone": str(data.get("phone", "N/A")),
            "location": str(data.get("location", "N/A")),
            "current_role": str(data.get("current_role", "N/A")),
            "years_experience": int(data.get("years_experience", 0)) if isinstance(data.get("years_experience"), (int, float)) else 0,
            "previous_companies": data.get("previous_companies", [])[:5] if isinstance(data.get("previous_companies"), list) else [],
            "skills": data.get("skills", [])[:15] if isinstance(data.get("skills"), list) else [],
            "certifications": data.get("certifications", [])[:10] if isinstance(data.get("certifications"), list) else [],
            "degree": str(data.get("degree", "N/A")),
            "university": str(data.get("university", "N/A")),
            "graduation_year": str(data.get("graduation_year", "N/A")),
            "uploaded_at": datetime.now().isoformat()
        }
        
        embedding_text = f"{resume_info['current_role']} {' '.join(resume_info['skills'])} {resume_info['name']}"
        embedding = embeddings.embed_query(embedding_text)
        
        point = PointStruct(
            id=file_id,
            vector=embedding,
            payload=resume_info
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])
        
        print(f"✅ {resume_info['name']} - {resume_info['current_role']}")
        return resume_info
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return None

# ============== API ENDPOINTS ==============

@app.get("/")
async def root():
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    return {
        "message": "LangGraph Multi-Agent Resume Intelligence - PRODUCTION",
        "architecture": "Intelligent Query Understanding with Dynamic Routing",
        "agents": [
            "Query Understanding Agent (Intent Detection)",
            "Details Agent (Get Specific Candidate)",
            "List Agent (Show All Candidates)",
            "Stats Agent (Database Statistics)",
            "Search Agent (Semantic Discovery)",
            "Analysis Agent (Deep Reasoning)",
            "Comparison Agent (Side-by-Side)",
            "Ranking Agent (Final Decision)"
        ],
        "supported_queries": [
            "Find candidates: 'find Python developers', 'who knows AWS'",
            "Get details: 'tell me about John', 'what are Dexter's skills'",
            "Compare: 'compare John and Sarah'",
            "List all: 'show all resumes', 'list everyone'",
            "Statistics: 'show stats', 'how many resumes'",
            "Count: 'how many Python developers'"
        ],
        "stats": {
            "total_resumes": collection_info.points_count
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat through intelligent multi-agent system"""
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    print(f"\n{'='*70}")
    print(f"🚀 PRODUCTION LANGGRAPH SYSTEM")
    print(f"Query: {request.message}")
    print(f"{'='*70}")
    
    try:
        initial_state = AgentState(
            messages=[HumanMessage(content=request.message)],
            user_query=request.message,
            query_intent="",
            target_names=[],
            search_results=[],
            analysis_results={},
            comparison_results={},
            final_recommendation="",
            next_agent="query_understanding",
            iteration_count=0,
            agent_decisions=[],
            execution_log=[]
        )
        
        final_state = resume_graph.invoke(initial_state)
        
        if final_state.get("final_recommendation"):
            response = final_state["final_recommendation"]
        else:
            response = "I couldn't process your query. Please try rephrasing."
        
        agent_flow = []
        for decision in final_state.get("agent_decisions", []):
            agent_name = decision.split(":")[0].strip()
            if agent_name not in agent_flow:
                agent_flow.append(agent_name)
        
        print(f"\n{'='*70}")
        print(f"✅ WORKFLOW COMPLETE")
        print(f"Intent: {final_state.get('query_intent', 'UNKNOWN')}")
        print(f"Agent Flow: {' → '.join(agent_flow)}")
        print(f"{'='*70}\n")
        
        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            agent_flow=agent_flow,
            execution_details={
                "query_intent": final_state.get("query_intent", "UNKNOWN"),
                "target_names": final_state.get("target_names", []),
                "agents_involved": len(agent_flow),
                "iterations": final_state.get("iteration_count", 0),
                "candidates_analyzed": len(final_state.get("search_results", [])),
                "execution_log": final_state.get("execution_log", [])
            }
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return ChatResponse(
            response=f"I encountered an error: {str(e)}. Please try again or rephrase your question.",
            conversation_id=conversation_id,
            agent_flow=["Error Handler"],
            execution_details={"error": str(e)}
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_resumes(files: List[UploadFile] = File(...)):
    """Upload and process resumes"""
    print(f"\n📤 Uploading {len(files)} resumes...")
    
    processed_resumes = []
    
    for file in files:
        content = await file.read()
        extension = os.path.splitext(file.filename)[1]
        
        resume = await process_resume_simple(content, file.filename, extension)
        if resume:
            processed_resumes.append(resume)
    
    print(f"✅ Processed {len(processed_resumes)}/{len(files)} resumes\n")
    
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
            "architecture": "Intelligent Query Understanding",
            "agents": 8,
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
        return {"message": "All resumes cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "Production LangGraph Multi-Agent",
        "agents": [
            "Query Understanding",
            "Details",
            "List",
            "Stats",
            "Search",
            "Analysis",
            "Comparison",
            "Ranking"
        ]
    }

@app.on_event("startup")
async def startup():
    print("\n" + "="*70)
    print("🚀 PRODUCTION LANGGRAPH MULTI-AGENT SYSTEM")
    print("="*70)
    print("Architecture: Intelligent Query Understanding")
    print("\nAgent Hierarchy:")
    print("   🧩 Query Understanding Agent (Intent Detection)")
    print("      ├── 📋 Details Agent (Specific Candidates)")
    print("      ├── 📝 List Agent (All Candidates)")
    print("      ├── 📊 Stats Agent (Statistics)")
    print("      └── 🔍 Search Flow:")
    print("          ├── 🔍 Search Agent")
    print("          ├── 🧠 Analysis Agent")
    print("          ├── ⚖️ Comparison Agent")
    print("          └── 🏆 Ranking Agent")
    print("\n" + "="*70)
    print("✨ SUPPORTED QUERIES:")
    print("   • 'find Python developers'")
    print("   • 'tell me about Dexter'")
    print("   • 'what are John's skills?'")
    print("   • 'compare Sarah and Mike'")
    print("   • 'show all resumes'")
    print("   • 'how many candidates?'")
    print("   • 'list everyone'")
    print("="*70)
    print(f"🌐 Server: http://127.0.0.1:8000")
    print(f"📊 Qdrant: {QDRANT_URL}")
    print(f"🤖 LLM: {OLLAMA_MODEL}")
    print("="*70 + "\n")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")