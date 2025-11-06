"""
Arytic - Intelligent Resume RAG System with Chain-of-Thought Reasoning
Version 7.0 - CoT + RAG for deeper analysis
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

# SpaCy for NER
try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.pipeline import EntityRuler
    SPACY_AVAILABLE = True
    print("✅ spaCy available for NER")
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️ spaCy not available - using LLM-only extraction")

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
app = FastAPI(title="Arytic - CoT + RAG Resume System")

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

# Initialize spaCy NER model
nlp = None
if SPACY_AVAILABLE:
    try:
        # Try to load transformer model first (best accuracy)
        try:
            nlp = spacy.load("en_core_web_trf")
            print("✅ Loaded spaCy transformer model (en_core_web_trf)")
        except:
            # Fallback to large model
            try:
                nlp = spacy.load("en_core_web_lg")
                print("✅ Loaded spaCy large model (en_core_web_lg)")
            except:
                # Fallback to small model
                nlp = spacy.load("en_core_web_sm")
                print("✅ Loaded spaCy small model (en_core_web_sm)")
        
        # Add entity ruler for better location/skill extraction
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            
            # Add location patterns
            patterns = [
                {"label": "GPE", "pattern": [{"LOWER": "north"}, {"LOWER": "carolina"}]},
                {"label": "GPE", "pattern": [{"TEXT": "NC"}]},
                {"label": "GPE", "pattern": [{"LOWER": "chapel"}, {"LOWER": "hill"}]},
                {"label": "GPE", "pattern": [{"TEXT": "UNC"}]},
            ]
            ruler.add_patterns(patterns)
            print("✅ Added entity ruler for enhanced extraction")
    except Exception as e:
        print(f"⚠️ Could not initialize spaCy: {e}")
        nlp = None

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
    reasoning_chain: List[Dict[str, str]]  # NEW: CoT reasoning steps

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
    reasoning_chain: List[Dict[str, str]]  # NEW: Track CoT reasoning

# ============== CHAIN-OF-THOUGHT UTILITIES ==============

class ChainOfThoughtEngine:
    """Handles explicit reasoning steps for transparent decision making"""
    
    @staticmethod
    def parse_cot_response(response_text: str) -> Dict[str, Any]:
        """Extract structured reasoning from CoT response"""
        reasoning_steps = []
        conclusion = ""
        
        # Extract thinking steps
        thinking_pattern = r"(?:Step \d+|First|Second|Third|Finally|Therefore):\s*(.+?)(?=(?:Step \d+|First|Second|Third|Finally|Therefore|$))"
        steps = re.findall(thinking_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        for step in steps:
            step_clean = step.strip()
            if step_clean:
                reasoning_steps.append(step_clean)
        
        # Extract conclusion
        conclusion_pattern = r"(?:Conclusion|Final Answer|Result):\s*(.+?)(?=\n\n|\Z)"
        conclusion_match = re.search(conclusion_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if conclusion_match:
            conclusion = conclusion_match.group(1).strip()
        
        return {
            "reasoning_steps": reasoning_steps,
            "conclusion": conclusion,
            "full_response": response_text
        }

# ============== INTELLIGENT EXTRACTION WITH CoT ==============

class IntelligentExtractionAgent:
    """Hybrid extraction: spaCy NER + LLM for comprehensive extraction"""
    
    def __init__(self, llm, spacy_nlp=None):
        self.llm = llm
        self.nlp = spacy_nlp
        self.name = "🤖 Intelligent Extraction Agent (Hybrid: spaCy NER + LLM CoT)"
    
    def extract_with_spacy(self, text: str) -> dict:
        """Fast extraction using spaCy NER"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(text[:10000])  # Process first 10k chars
        
        extracted = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "emails": [],
            "phones": []
        }
        
        # Extract entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                extracted["persons"].append(ent.text)
            elif ent.label_ == "ORG":
                extracted["organizations"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:  # Geopolitical entities and locations
                extracted["locations"].append(ent.text)
        
        # Extract email using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        extracted["emails"] = emails[:3]
        
        # Extract phone using regex
        phone_pattern = r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        extracted["phones"] = phones[:3]
        
        return extracted
    
    def extract_resume_data(self, text: str, filename: str) -> dict:
        """Use LLM with CoT to intelligently extract resume fields"""
        
        print(f"\n   🤖 {self.name}: Analyzing resume with reasoning")
        
        # Truncate text to fit context
        resume_text = text[:4000]
        
        extraction_prompt = f"""You are an expert resume parser. Use step-by-step reasoning to extract information.

RESUME TEXT:
{resume_text}

Use Chain-of-Thought reasoning to extract information:

**Step 1: Identify Document Structure**
- Think: Where is the header? Where are sections like Experience, Skills, Education?

**Step 2: Extract Contact Information**
- Think: Look for name (usually top/header), email (pattern: x@y.com), phone (10 digits)

**Step 3: Determine Experience Level**
- Think: Calculate years from dates OR look for "X years experience" statements

**Step 4: Identify Skills and Certifications**
- Think: Look for dedicated sections, bullet points with technical terms

**Step 5: Extract Education AND Location**
- Think: Find degree type, university name, graduation year
- CRITICAL FOR LOCATION: 
  * If university contains "University of [STATE]" → extract STATE as location
  * "University of North Carolina" → location = "North Carolina"
  * "University of Texas" → location = "Texas"  
  * "Chapel Hill" → location = "North Carolina" (commonly known)
  * If city is mentioned, include both city and state
- Also look for current address in header/contact section

**LOCATION EXTRACTION EXAMPLES:**
- "Masters at University of North Carolina Chapel Hill" → location: "North Carolina", city: "Chapel Hill"
- "Client: Google, Mountain View, CA" → location: "California", city: "Mountain View"
- "McLean, VA" → location: "Virginia", city: "McLean"
- "Chicago, IL" → location: "Illinois", city: "Chicago"

Now provide extraction in JSON format:

{{
  "reasoning": {{
    "name_reasoning": "Found name in header at top",
    "experience_reasoning": "Calculated 5 years from 2019-2024 in work history",
    "skills_reasoning": "Identified 15 skills in Technical Skills section"
  }},
  "extracted_data": {{
    "name": "Full Name",
    "email": "email@domain.com",
    "phone": "123-456-7890",
    "location": "City, State",
    "current_role": "Job Title",
    "years_experience": 5,
    "skills": ["skill1", "skill2"],
    "certifications": ["cert1"],
    "education": {{
      "degree": "Bachelor's",
      "university": "University Name",
      "graduation_year": "2020"
    }},
    "experience": [
      {{
        "company": "Company",
        "role": "Role",
        "duration": "2020-2024",
        "description": "Brief description"
      }}
    ]
  }}
}}

Return valid JSON only:"""

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
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                
                data = json.loads(json_str)
                extracted = data.get('extracted_data', data)
                
                # Validate and structure data
                final_data = {
                    "name": str(extracted.get('name', 'N/A')).strip(),
                    "email": str(extracted.get('email', 'N/A')).strip(),
                    "phone": str(extracted.get('phone', 'N/A')).strip(),
                    "location": str(extracted.get('location', 'N/A')).strip(),
                    "current_role": str(extracted.get('current_role', 'N/A')).strip(),
                    "years_experience": self._parse_years(extracted.get('years_experience', 0)),
                    "skills": self._parse_list(extracted.get('skills', [])),
                    "certifications": self._parse_list(extracted.get('certifications', [])),
                    "degree": "N/A",
                    "university": "N/A",
                    "graduation_year": "N/A",
                    "experience": []
                }
                
                # Parse education
                if isinstance(extracted.get('education'), dict):
                    edu = extracted['education']
                    final_data['degree'] = str(edu.get('degree', 'N/A')).strip()
                    final_data['university'] = str(edu.get('university', 'N/A')).strip()
                    final_data['graduation_year'] = str(edu.get('graduation_year', 'N/A')).strip()
                
                # Parse experience
                if isinstance(extracted.get('experience'), list):
                    final_data['experience'] = extracted['experience'][:5]
                
                # Create companies list
                final_data['previous_companies'] = [
                    exp.get('company', 'N/A') 
                    for exp in final_data['experience'] 
                    if exp.get('company')
                ][:5]
                
                print(f"   ✅ {final_data['name']} | {final_data['current_role']}")
                print(f"   ✅ Skills: {len(final_data['skills'])} | Certs: {len(final_data['certifications'])}")
                
                return final_data
            else:
                return self.create_fallback_data(text, filename)
                
        except Exception as e:
            print(f"   ⚠️ Extraction error: {e}")
            return self.create_fallback_data(text, filename)
    
    def _parse_years(self, value) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r'(\d+)', value)
            if match:
                return int(match.group(1))
        return 0
    
    def _parse_list(self, value) -> list:
        if isinstance(value, list):
            return [str(item).strip() for item in value if item][:30]
        return []
    
    def _enhance_locations(self, resume_data: dict, full_text: str) -> dict:
        """Post-process to extract locations that LLM might have missed"""
        
        all_locations = set(resume_data.get('all_locations', []))
        
        # Common university -> state mappings
        university_states = {
            'University of North Carolina': 'North Carolina',
            'UNC': 'North Carolina',
            'Chapel Hill': 'North Carolina',
            'University of California': 'California',
            'UC Berkeley': 'California',
            'University of Texas': 'Texas',
            'MIT': 'Massachusetts',
            'University of Wisconsin': 'Wisconsin',
            'University of Illinois': 'Illinois',
            'University of Michigan': 'Michigan',
            'University of Washington': 'Washington',
            'University of Virginia': 'Virginia',
            'University of Maryland': 'Maryland',
            'University of Georgia': 'Georgia',
            'University of Florida': 'Florida',
        }
        
        # Check education entries for locations
        for edu in resume_data.get('education', []):
            uni = edu.get('university', '')
            
            # Check university name against known mappings
            for uni_pattern, state in university_states.items():
                if uni_pattern.lower() in uni.lower():
                    all_locations.add(state)
                    if edu.get('location', 'N/A') == 'N/A':
                        edu['location'] = state
            
            # Add explicit education location
            if edu.get('location') and edu.get('location') != 'N/A':
                all_locations.add(edu['location'])
        
        # Extract "City, STATE" patterns from full text
        city_state_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z]{2})\b'
        matches = re.findall(city_state_pattern, full_text)
        
        # State abbreviation to full name
        state_abbrev = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming'
        }
        
        for city, state_abbr_code in matches:
            all_locations.add(city)
            if state_abbr_code in state_abbrev:
                all_locations.add(state_abbrev[state_abbr_code])
        
        # Extract experience locations
        for exp in resume_data.get('experience', []):
            if exp.get('location') and exp.get('location') != 'N/A':
                # Parse "City, State" or "City, ST"
                loc = exp['location']
                if ',' in loc:
                    parts = loc.split(',')
                    all_locations.add(parts[0].strip())  # City
                    state_part = parts[1].strip()
                    if state_part in state_abbrev:
                        all_locations.add(state_abbrev[state_part])
                    else:
                        all_locations.add(state_part)
                else:
                    all_locations.add(loc)
        
        # Update resume data
        resume_data['all_locations'] = list(all_locations)
        
        # Set current_location if still N/A
        if resume_data.get('current_location', 'N/A') == 'N/A' and all_locations:
            # Try to get from most recent experience first
            if resume_data.get('experience') and resume_data['experience'][0].get('location'):
                resume_data['current_location'] = resume_data['experience'][0]['location']
            elif resume_data.get('education') and resume_data['education'][0].get('location'):
                resume_data['current_location'] = resume_data['education'][0]['location']
            elif all_locations:
                resume_data['current_location'] = list(all_locations)[0]
        
        print(f"   🔧 Enhanced locations: {', '.join(list(all_locations)[:5])}")
        
        return resume_data
    
    def create_fallback_data(self, text: str, filename: str) -> dict:
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        name = filename.replace('.pdf', '').replace('.docx', '').replace('_', ' ').strip().title()
        
        return {
            "name": name,
            "email": email_match.group(0) if email_match else "N/A",
            "phone": phone_match.group(0) if phone_match else "N/A",
            "current_location": "N/A",
            "all_locations": [],
            "current_role": "N/A",
            "years_experience": 0,
            "skills": [],
            "certifications": [],
            "education": [],
            "experience": [],
            "previous_companies": [],
            "highest_degree": "N/A",
            "primary_university": "N/A"
        }


# ============== AGENT 1: QUERY UNDERSTANDING WITH CoT ==============
class QueryUnderstandingAgent:
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "🧠 Query Understanding Agent (CoT)"
    
    def __call__(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        
        log_msg = f"{self.name}: Analyzing query with reasoning"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        understanding_prompt = f"""Analyze this hiring query using step-by-step reasoning.

QUERY: "{query}"

Use Chain-of-Thought reasoning:

**Step 1: Identify explicit requirements**
Think: What job title, skills, experience level, or location are directly mentioned?

**Step 2: Infer implicit requirements**
Think: What related skills might be important? What experience level makes sense?

**Step 3: Prioritize requirements**
Think: Which requirements are must-haves vs nice-to-haves?

**Step 4: Formulate search strategy**
Think: What keywords will find the best matches?

Now provide structured output:

{{
  "reasoning": {{
    "analysis": "I identified the role as X because...",
    "key_insights": "The user prioritizes skills Y and Z because...",
    "search_strategy": "I will search for keywords A, B, C because..."
  }},
  "requirements": {{
    "role": "job title or null",
    "skills": ["required skills"],
    "experience_years": "number or null",
    "location": "location or null",
    "certifications": ["certifications"],
    "keywords": ["other important keywords"],
    "priority_level": "high/medium/low for each requirement"
  }}
}}

Return valid JSON only:"""

        try:
            result = self.llm.invoke(understanding_prompt)
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            # Parse CoT response
            cot_parsed = ChainOfThoughtEngine.parse_cot_response(result_text)
            
            # Clean and parse JSON
            result_text = re.sub(r'```json\s*', '', result_text)
            result_text = re.sub(r'```\s*', '', result_text)
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            
            if json_match:
                json_str = json_match.group(0)
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                data = json.loads(json_str)
                
                requirements = data.get('requirements', data)
                reasoning = data.get('reasoning', {})
                
                state["parsed_requirements"] = requirements
                
                # Add reasoning to chain
                state["reasoning_chain"].append({
                    "agent": self.name,
                    "step": "Query Analysis",
                    "reasoning": str(reasoning),
                    "output": str(requirements)
                })
                
                log_msg = f"✓ Requirements: {requirements}"
                print(f"   {log_msg}")
                state["execution_log"].append(log_msg)
                
                if reasoning:
                    print(f"   💭 Reasoning: {reasoning.get('analysis', '')[:100]}...")
            else:
                state["parsed_requirements"] = {}
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            state["parsed_requirements"] = {}
        
        state["next_agent"] = "search"
        return state


# ============== AGENT 2: INTELLIGENT SEARCH WITH CoT ==============
class IntelligentSearchAgent:
    
    def __init__(self, qdrant_client, embeddings):
        self.qdrant = qdrant_client
        self.embeddings = embeddings
        self.name = "🔍 Intelligent Search Agent (CoT)"
    
    def __call__(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        requirements = state.get("parsed_requirements", {})
        
        log_msg = f"{self.name}: Searching with reasoning"
        print(f"\n{log_msg}")
        state["execution_log"].append(log_msg)
        
        # CoT: Explain search strategy
        search_reasoning = {
            "strategy": "Semantic search using embeddings",
            "query_construction": [],
            "expected_matches": "Candidates with relevant skills and experience"
        }
        
        # Build semantic search query with reasoning
        search_parts = []
        if requirements.get('role'):
            search_parts.append(requirements['role'])
            search_reasoning["query_construction"].append(f"Added role: {requirements['role']}")
        if requirements.get('skills'):
            search_parts.extend(requirements['skills'])
            search_reasoning["query_construction"].append(f"Added skills: {requirements['skills']}")
        if requirements.get('location'):
            # CRITICAL: Include location in semantic search
            search_parts.append(requirements['location'])
            search_reasoning["query_construction"].append(f"Added location: {requirements['location']}")
        if requirements.get('keywords'):
            search_parts.extend(requirements['keywords'])
        if requirements.get('certifications'):
            search_parts.extend(requirements['certifications'])
            search_reasoning["query_construction"].append(f"Added certs: {requirements['certifications']}")
        
        search_query = ' '.join(search_parts) if search_parts else query
        
        log_msg = f"Search query: {search_query}"
        print(f"   {log_msg}")
        state["execution_log"].append(log_msg)
        
        # Semantic search
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
        
        # Add search reasoning to chain
        state["reasoning_chain"].append({
            "agent": self.name,
            "step": "Candidate Search",
            "reasoning": str(search_reasoning),
            "output": f"Found {len(candidates)} candidates via semantic search"
        })
        
        log_msg = f"✓ Found {len(candidates)} candidates"
        print(f"   {log_msg}")
        state["execution_log"].append(log_msg)
        
        state["next_agent"] = "analysis"
        return state


# ============== AGENT 3: INTELLIGENT MATCHING WITH DEEP CoT ==============
class IntelligentMatchingAgent:
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "🎯 Intelligent Matching Agent (Deep CoT)"
    
    def __call__(self, state: AgentState) -> AgentState:
        candidates = state["search_results"][:TOP_CANDIDATES]
        requirements = state.get("parsed_requirements", {})
        query = state["user_query"]
        
        log_msg = f"{self.name}: Deep reasoning analysis"
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
                "current_location": c.get('current_location'),
                "all_locations": c.get('all_locations', []),
                "education": c.get('education', []),
                "companies": c.get('previous_companies', [])[:3]
            }
            candidate_summaries.append(summary)
        
        matching_prompt = f"""You are an expert recruiter. Use detailed Chain-of-Thought reasoning to analyze candidates.

USER QUERY: "{query}"

REQUIREMENTS:
{json.dumps(requirements, indent=2)}

CANDIDATES:
{json.dumps(candidate_summaries, indent=2)}

For EACH candidate, use this reasoning process:

**Step 1: Skills Analysis**
Think: Which skills match? Which are missing? How critical are the gaps?

**Step 2: Experience Evaluation**
Think: Does experience level match requirements? Is the background relevant?

**Step 3: Cultural & Location Fit**
Think: Does location EXACTLY match requirements? 
- Check if candidate's CURRENT_LOCATION matches the requirement
- Check if ANY location in ALL_LOCATIONS matches the requirement (they studied or worked there)
- Check education locations - if they went to "University of [State]", they likely have connections there
- If location was specified as required, penalize heavily for NO matches across all locations
- Partial match: If they studied/worked there but aren't currently there, give partial credit (50% of location points)
- Full match: If current_location matches OR they have strong ties (multiple entries in all_locations), give full credit

**Step 4: Certification Assessment**
Think: Are required certifications present? What's missing?

**Step 5: Overall Scoring**
Think: Weigh all factors. Calculate final score with justification.

Return JSON array:
[
  {{
    "id": 0,
    "reasoning": {{
      "skills_analysis": "Has Python, Java. Missing React (required). 70% match.",
      "experience_analysis": "5 years matches requirement. Relevant background in SaaS.",
      "location_analysis": "Remote-friendly, location not a barrier.",
      "certification_analysis": "Has AWS, missing PMP (nice-to-have).",
      "scoring_logic": "Strong technical skills (40pts) + good experience (28pts) + location ok (15pts) + partial certs (10pts) = 93/100"
    }},
    "score": 93,
    "matched_skills": ["Python", "Java", "AWS"],
    "missing_skills": ["React"],
    "why_match": "Strong technical foundation with relevant SaaS experience",
    "concerns": ["Missing React experience"],
    "confidence": "high"
  }}
]

Return valid JSON array only:"""

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
                        candidate['missing_skills'] = analysis.get('missing_skills', [])
                        candidate['why_match'] = analysis.get('why_match', '')
                        candidate['concerns'] = analysis.get('concerns', [])
                        candidate['reasoning'] = analysis.get('reasoning', {})
                        candidate['confidence'] = analysis.get('confidence', 'medium')
                    else:
                        candidate['match_score'] = 50
                
                # Sort by score
                candidates.sort(key=lambda x: x.get('match_score', 0), reverse=True)
                
                state["analyzed_candidates"] = candidates
                
                # Add detailed reasoning to chain
                for i, candidate in enumerate(candidates[:3]):
                    state["reasoning_chain"].append({
                        "agent": self.name,
                        "step": f"Candidate {i+1} Analysis: {candidate.get('name')}",
                        "reasoning": str(candidate.get('reasoning', {})),
                        "output": f"Score: {candidate.get('match_score')}/100 - {candidate.get('why_match', 'N/A')}"
                    })
                
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
        
        # Create rich embedding
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
        "message": "Arytic - CoT + RAG Resume System",
        "version": "7.0",
        "features": [
            "✅ Chain-of-Thought reasoning at every stage",
            "✅ Explicit reasoning steps for transparency",
            "✅ Deep analysis with step-by-step logic",
            "✅ RAG with semantic search",
            "✅ Confidence scoring",
            "✅ Reasoning chain tracking"
        ],
        "stats": {
            "total_resumes": collection_info.points_count,
            "status": "operational"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process hiring queries with CoT reasoning"""
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
            execution_log=[],
            reasoning_chain=[]  # Initialize reasoning chain
        )
        
        final_state = arytic_graph.invoke(initial_state)
        
        candidates = final_state.get("analyzed_candidates", [])
        requirements = final_state.get("parsed_requirements", {})
        reasoning_chain = final_state.get("reasoning_chain", [])
        
        if not candidates:
            response = "❌ No matching candidates found."
        else:
            response = generate_cot_response(request.message, requirements, candidates, reasoning_chain)
        
        print(f"\n{'='*80}")
        print(f"✅ Found {len(candidates)} candidates with {len(reasoning_chain)} reasoning steps")
        print(f"{'='*80}\n")
        
        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            execution_details={
                "requirements": requirements,
                "candidates_found": len(candidates),
                "execution_log": final_state.get("execution_log", [])
            },
            reasoning_chain=reasoning_chain
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return ChatResponse(
            response=f"Error: {str(e)}",
            conversation_id=conversation_id,
            execution_details={"error": str(e)},
            reasoning_chain=[]
        )


def generate_cot_response(query: str, requirements: dict, candidates: list, reasoning_chain: list) -> str:
    """Generate detailed response with CoT reasoning"""
    
    response = f"🧠 CHAIN-OF-THOUGHT ANALYSIS\n{'='*80}\n\n"
    response += f"📝 Query: \"{query}\"\n\n"
    
    # Show reasoning chain
    if reasoning_chain:
        response += "💭 REASONING PROCESS:\n"
        response += f"{'-'*80}\n"
        for i, step in enumerate(reasoning_chain[:5], 1):  # Show first 5 steps
            response += f"\n{i}. {step.get('step', 'Unknown Step')}\n"
            reasoning_text = step.get('reasoning', '')
            if isinstance(reasoning_text, str) and len(reasoning_text) > 200:
                reasoning_text = reasoning_text[:200] + "..."
            response += f"   💡 {reasoning_text}\n"
            if step.get('output'):
                response += f"   ✓ Result: {step.get('output', '')}\n"
        response += f"\n{'-'*80}\n\n"
    
    if requirements:
        response += "🎯 EXTRACTED REQUIREMENTS:\n"
        for key, value in requirements.items():
            if value and key != 'priority_level':
                response += f"   • {key}: {value}\n"
        response += "\n"
    
    response += f"👥 TOP {min(len(candidates), TOP_CANDIDATES)} CANDIDATES:\n\n"
    
    for i, c in enumerate(candidates[:TOP_CANDIDATES], 1):
        response += f"{'='*80}\n"
        response += f"#{i} - {c.get('name', 'N/A')}\n"
        response += f"{'='*80}\n"
        response += f"🎯 Match Score: {c.get('match_score', 0)}/100"
        if c.get('confidence'):
            response += f" | Confidence: {c.get('confidence', 'N/A').upper()}\n\n"
        else:
            response += "\n\n"
        
        # Show CoT reasoning for this candidate
        if c.get('reasoning') and isinstance(c.get('reasoning'), dict):
            response += "💭 DETAILED REASONING:\n"
            reasoning = c.get('reasoning')
            
            if reasoning.get('skills_analysis'):
                response += f"   🛠️  Skills: {reasoning['skills_analysis']}\n"
            if reasoning.get('experience_analysis'):
                response += f"   📊 Experience: {reasoning['experience_analysis']}\n"
            if reasoning.get('location_analysis'):
                response += f"   📍 Location: {reasoning['location_analysis']}\n"
            if reasoning.get('certification_analysis'):
                response += f"   📜 Certifications: {reasoning['certification_analysis']}\n"
            if reasoning.get('scoring_logic'):
                response += f"   🎯 Scoring: {reasoning['scoring_logic']}\n"
            response += "\n"
        
        response += "💼 Profile:\n"
        response += f"   • Role: {c.get('current_role', 'N/A')}\n"
        response += f"   • Experience: {c.get('years_experience', 0)} years\n"
        response += f"   • Current Location: {c.get('current_location', 'N/A')}\n"
        
        # Show all locations if available
        if c.get('all_locations') and len(c.get('all_locations', [])) > 0:
            response += f"   • Location History: {', '.join(c.get('all_locations', [])[:5])}\n"
        
        # Show education with locations
        if c.get('education') and len(c.get('education', [])) > 0:
            response += f"   • Education:\n"
            for edu in c.get('education', [])[:2]:
                edu_str = f"     - {edu.get('degree', 'N/A')}"
                if edu.get('university') != 'N/A':
                    edu_str += f" - {edu.get('university')}"
                if edu.get('location') and edu.get('location') != 'N/A':
                    edu_str += f" ({edu.get('location')})"
                if edu.get('graduation_year') and edu.get('graduation_year') != 'N/A':
                    edu_str += f" - {edu.get('graduation_year')}"
                response += edu_str + "\n"
        elif c.get('highest_degree'):
            response += f"   • Education: {c.get('highest_degree', 'N/A')}"
            if c.get('primary_university') and c.get('primary_university') != 'N/A':
                response += f" - {c.get('primary_university')}\n"
            else:
                response += "\n"
        
        if c.get('previous_companies'):
            response += f"   • Companies: {', '.join(c.get('previous_companies', [])[:3])}\n"
        response += "\n"
        
        if c.get('matched_skills'):
            response += "✅ Matched Skills:\n"
            response += f"   {', '.join(c.get('matched_skills', []))}\n\n"
        
        if c.get('missing_skills'):
            response += "⚠️  Missing Skills:\n"
            response += f"   {', '.join(c.get('missing_skills', []))}\n\n"
        
        if c.get('skills'):
            response += "🛠️  All Skills:\n"
            skills = c.get('skills', [])[:15]
            response += f"   {', '.join(skills)}\n"
            if len(c.get('skills', [])) > 15:
                response += f"   ... and {len(c.get('skills', [])) - 15} more\n"
            response += "\n"
        
        if c.get('certifications'):
            response += "📜 Certifications:\n"
            response += f"   • {', '.join(c.get('certifications', []))}\n\n"
        
        if c.get('why_match'):
            response += "💡 Why Good Match:\n"
            response += f"   {c.get('why_match')}\n\n"
        
        if c.get('concerns'):
            response += "⚠️  Considerations:\n"
            for concern in c.get('concerns', []):
                response += f"   • {concern}\n"
            response += "\n"
        
        response += "📧 Contact:\n"
        response += f"   • Email: {c.get('email', 'N/A')}\n"
        response += f"   • Phone: {c.get('phone', 'N/A')}\n\n"
    
    # Top recommendation with reasoning
    if candidates:
        top = candidates[0]
        response += f"{'='*80}\n"
        response += "🏆 TOP RECOMMENDATION\n"
        response += f"{'='*80}\n\n"
        response += f"**{top.get('name')}** - Score: {top.get('match_score', 0)}/100"
        if top.get('confidence'):
            response += f" | {top.get('confidence', '').upper()} confidence\n\n"
        else:
            response += "\n\n"
        
        if top.get('why_match'):
            response += f"**Why?** {top.get('why_match')}\n\n"
        
        if top.get('reasoning') and isinstance(top.get('reasoning'), dict):
            reasoning = top.get('reasoning')
            if reasoning.get('scoring_logic'):
                response += f"**Scoring Logic:** {reasoning['scoring_logic']}\n\n"
        
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
            "architecture": "CoT + RAG (3 Agents with Chain-of-Thought)",
            "model": OLLAMA_MODEL,
            "reasoning": "Enabled",
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
        "system": "Arytic CoT + RAG",
        "version": "7.0"
    }

@app.on_event("startup")
async def startup():
    print("\n" + "="*80)
    print("🚀 ARYTIC - CHAIN-OF-THOUGHT + RAG RESUME SYSTEM")
    print("="*80)
    
    print("\n⚡ KEY FEATURES:")
    print(f"   ✅ Chain-of-Thought (CoT) reasoning at every stage")
    print(f"   ✅ Explicit step-by-step analysis for transparency")
    print(f"   ✅ Deep reasoning with justification for decisions")
    print(f"   ✅ RAG with semantic search via embeddings")
    print(f"   ✅ Confidence scoring for matches")
    print(f"   ✅ Complete reasoning chain tracking")
    
    print(f"\n🧠 REASONING PIPELINE:")
    print(f"   1️⃣  Query Understanding Agent - CoT requirement extraction")
    print(f"   2️⃣  Intelligent Search Agent - CoT search strategy")
    print(f"   3️⃣  Intelligent Matching Agent - Deep CoT analysis")
    print(f"       • Skills analysis with gap identification")
    print(f"       • Experience evaluation with relevance check")
    print(f"       • Location & cultural fit assessment")
    print(f"       • Certification validation")
    print(f"       • Weighted scoring with full justification")
    
    print(f"\n📊 CoT BENEFITS:")
    print(f"   • Transparent decision making")
    print(f"   • Explainable AI - see the reasoning")
    print(f"   • Better accuracy through step-by-step logic")
    print(f"   • Confidence metrics for each match")
    print(f"   • Audit trail of all decisions")
    
    print(f"\n🔧 Configuration:")
    print(f"   • Model: {OLLAMA_MODEL}")
    print(f"   • Temperature: 0.1 (precise reasoning)")
    print(f"   • CPU Threads: {CPU_THREADS}")
    print(f"   • Context: {CONTEXT_SIZE} tokens")
    print(f"   • Embedding: {EMBEDDING_MODEL}")
    
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
    