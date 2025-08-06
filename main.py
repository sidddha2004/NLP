# Phase 2: Query & Retrieval API (Deploy on Railway)
# This is a lightweight API that only handles queries and retrieval

import os
from typing import List
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI and related imports
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# AI and vector database - only query functionality
from pinecone import Pinecone
import google.generativeai as genai

# FastAPI setup
app = FastAPI(title="Insurance Policy RAG Query API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
gemini_model = None
pc = None
index = None
initialization_status = {
    "gemini": False,
    "pinecone": False,
    "ready": False,
    "error": None
}

class QueryService:
    def __init__(self):
        self.gemini_model = None
        self.index = None
        
    def initialize_services(self):
        """Initialize Gemini and Pinecone services"""
        global gemini_model, pc, index, initialization_status
        
        try:
            # Initialize Gemini
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            self.gemini_model = gemini_model
            logger.info("✓ Gemini model initialized successfully")
            initialization_status["gemini"] = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            initialization_status["error"] = str(e)
            raise e
    
    def connect_to_pinecone(self):
        """Connect to existing Pinecone index"""
        global pc, index, initialization_status
        
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            pc = Pinecone(api_key=api_key)
            logger.info("✓ Pinecone client initialized")
            
            # Connect to existing index
            index_name = "policy-docs-gemini-hash"
            
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if index_name not in existing_indexes:
                raise Exception(f"Index '{index_name}' not found. Please run Phase 1 first to create and populate the index.")
            
            index = pc.Index(index_name)
            self.index = index
            logger.info(f"✓ Connected to existing index: {index_name}")
            
            # Verify index has data
            stats = index.describe_index_stats()
            if stats.total_vector_count == 0:
                raise Exception("Index is empty. Please run Phase 1 first to populate the index with document embeddings.")
            
            logger.info(f"✓ Index ready with {stats.total_vector_count} vectors")
            initialization_status["pinecone"] = True
            initialization_status["ready"] = True
            
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            initialization_status["error"] = str(e)
            raise e
    
    def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for query using EXACT same method as Phase 1"""
        try:
            # Use Gemini to enhance query understanding (same prompt style as Phase 1)
            prompt = f"""
            Extract key insurance concepts and terms from this question. Focus on:
            - Coverage types and limits
            - Policy conditions and terms  
            - Exclusions and restrictions
            - Claims procedures
            - Premium and payment details
            - Legal and regulatory terms
            
            Question: {query}
            
            List 30 most important keywords/concepts (comma-separated):
            """
            
            response = self.gemini_model.generate_content(prompt)
            keywords = response.text.strip()
            
            # Combine keywords with original query (same as Phase 1)
            enhanced_text = f"{keywords} {query}"
            embedding = self._create_hybrid_embedding(enhanced_text, query)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            # Fallback to hybrid embedding
            return self._create_hybrid_embedding(query, query)
    
    def _create_hybrid_embedding(self, enhanced_text: str, original_text: str) -> List[float]:
        """Create hybrid embedding - MUST match Phase 1 exactly"""
        # Method 1: Hash-based semantic features (384 dims)
        semantic_features = self._get_semantic_hash_features(enhanced_text, 384)
        
        # Method 2: Statistical text features (384 dims) 
        statistical_features = self._get_statistical_features(original_text, 384)
        
        # Combine both approaches
        embedding = semantic_features + statistical_features
        
        # Ensure exactly 768 dimensions
        embedding = embedding[:768]
        while len(embedding) < 768:
            embedding.append(0.0)
        
        return embedding
    
    def _get_semantic_hash_features(self, text: str, target_dims: int) -> List[float]:
        """Generate semantic hash features - IDENTICAL to Phase 1"""
        import hashlib
        features = []
        text = text.lower().strip()
        
        # Create semantic variations (same as Phase 1)
        variations = [
            text,
            ' '.join(sorted(text.split())),  # Sorted words
            ''.join(c for c in text if c.isalnum() or c.isspace()),  # Alphanumeric only
        ]
        
        for variation in variations:
            for i in range(target_dims // len(variations)):
                if len(features) >= target_dims:
                    break
                hash_val = hashlib.sha256(f"{variation}_{i}".encode()).digest()
                for byte_val in hash_val:
                    if len(features) >= target_dims:
                        break
                    features.append((byte_val / 255.0) * 2 - 1)
        
        return features[:target_dims]
    
    def _get_statistical_features(self, text: str, target_dims: int) -> List[float]:
        """Generate statistical text features - IDENTICAL to Phase 1"""
        features = []
        words = text.lower().split()
        
        # Basic statistics (same as Phase 1)
        basic_stats = [
            len(text) / 10000.0,  # Text length (normalized)
            len(words) / 1000.0,  # Word count (normalized)  
            len(set(words)) / max(len(words), 1),  # Unique word ratio
            sum(len(w) for w in words) / max(len(words), 1) / 15.0,  # Avg word length
            text.count('.') / max(len(text), 1) * 100,  # Sentence density
            text.count(',') / max(len(text), 1) * 100,  # Comma density
            sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
            sum(1 for c in text if c.isdigit()) / max(len(text), 1),  # Digit ratio
        ]
        
        features.extend(basic_stats)
        
        # Word-based features (same terms as Phase 1)
        common_insurance_terms = [
            'policy', 'coverage', 'premium', 'deductible', 'claim', 'benefit', 
            'exclusion', 'condition', 'insured', 'insurer', 'liability', 'limit'
        ]
        
        for term in common_insurance_terms:
            count = text.lower().count(term)
            features.append(min(count / max(len(words), 1) * 100, 1.0))
        
        # Character n-gram features (same as Phase 1)
        for n in [2, 3, 4]:
            ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
            for i in range(min(50, len(ngrams))):
                if len(features) >= target_dims:
                    break
                hash_val = hash(ngrams[i]) % 10000
                features.append(hash_val / 10000.0)
        
        # Pad to target dimensions
        while len(features) < target_dims:
            features.append(0.0)
        
        return features[:target_dims]
    
    def search_relevant_chunks(self, query: str, top_k: int = 8) -> List[dict]:
        """Search for relevant chunks with improved scoring"""
        try:
            query_embedding = self.create_query_embedding(query)
            
            if not query_embedding:
                return []
            
            # Search with higher top_k for better results
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k * 2,  # Get more candidates
                include_metadata=True
            )
            
            relevant_chunks = []
            seen_texts = set()  # Avoid duplicates
            
            for match in query_response.matches:
                # Lower similarity threshold for better recall
                if match.score > 0.1:  # Reduced from 0.3
                    text = match.metadata.get("text", "")
                    if text and text not in seen_texts:
                        relevant_chunks.append({
                            'text': text,
                            'score': match.score,
                            'length': match.metadata.get('length', len(text))
                        })
                        seen_texts.add(text)
                        
                        if len(relevant_chunks) >= top_k:
                            break
            
            # Sort by score descending
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks for query")
            if relevant_chunks:
                logger.info(f"Best match score: {relevant_chunks[0]['score']:.3f}")
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error searching for relevant chunks: {e}")
            return []
    
    def generate_answer(self, question: str, context_chunks: List[dict]) -> str:
        """Generate precise, policy-specific answers using enhanced prompting"""
        try:
            if not context_chunks:
                return "I couldn't find relevant information in the policy document to answer your question. Please try rephrasing your question or contact support for assistance."
            
            # Sort chunks by relevance score and take best ones
            sorted_chunks = sorted(context_chunks, key=lambda x: x['score'], reverse=True)
            best_chunks = sorted_chunks[:5]  # Use top 5 chunks
            
            # Create focused context
            context_parts = []
            for i, chunk in enumerate(best_chunks):
                context_parts.append(f"Policy Section {i+1}:\n{chunk['text']}")
            
            context = "\n\n" + "="*50 + "\n\n".join(context_parts)
            
            # Enhanced prompt for precise, policy-specific answers
            prompt = f"""
You are an expert insurance policy analyst specializing in providing precise, actionable answers from policy documents.

Question: {question}

Policy Context:
{context}

INSTRUCTIONS FOR RESPONSE FORMAT:
1. Start with a direct, specific answer to the question
2. Quote exact policy terms, periods, amounts, and conditions
3. Use precise language from the policy document
4. Include specific numbers, timeframes, and limits when mentioned
5. Be concise but comprehensive
6. If there are conditions or exceptions, state them clearly

RESPONSE STYLE:
- Use declarative statements (e.g., "A grace period of thirty days is provided...")
- Include specific details (amounts, timeframes, percentages)
- Quote policy language when relevant
- Mention eligibility criteria if applicable
- State limitations or exclusions clearly

EXAMPLES OF GOOD RESPONSES:
- "A grace period of thirty days is provided for premium payment after the due date..."
- "There is a waiting period of thirty-six (36) months of continuous coverage..."
- "The policy covers maternity expenses with a benefit limit of [amount] per delivery..."
- "Pre-existing diseases are covered after a waiting period of [X] months..."

If the context doesn't contain sufficient information to answer the question specifically, say: "The policy document provided does not contain specific information about [topic]. Please refer to the complete policy document or contact your insurance provider."

Generate a precise, policy-focused answer:
"""
            
            response = self.gemini_model.generate_content(prompt)
            answer = response.text.strip()
            
            # Post-process to ensure quality
            if len(answer) < 50 and "sufficient information" not in answer.lower():
                # If answer is too short and not an "insufficient info" response, try to enhance it
                enhanced_prompt = f"""
Based on this policy context, provide a more detailed answer to: {question}

Context: {context[:1000]}

Provide specific details including:
- Exact time periods mentioned
- Specific amounts or limits
- Eligibility requirements
- Any conditions or restrictions

Answer:
"""
                try:
                    enhanced_response = self.gemini_model.generate_content(enhanced_prompt)
                    if len(enhanced_response.text.strip()) > len(answer):
                        answer = enhanced_response.text.strip()
                except:
                    pass  # Keep original answer if enhancement fails
            
            # Add debugging info in logs
            logger.info(f"Generated answer length: {len(answer)} chars")
            scores_list = [f"{c['score']:.3f}" for c in best_chunks]
            logger.info(f"Used {len(best_chunks)} context chunks with scores: {scores_list}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while processing your question: {str(e)}. Please try again or contact support."

# Initialize service
query_service = QueryService()

# Authentication
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    expected_token = os.getenv("API_BEARER_TOKEN")
    
    if not expected_token:
        logger.error("API_BEARER_TOKEN not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_BEARER_TOKEN not configured"
        )
    if token != expected_token:
        logger.warning("Invalid token received")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid authentication token"
        )
    return True

# Request/Response models
class QueryRequest(BaseModel):
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("=== PHASE 2 API STARTUP ===")
    
    try:
        # Initialize services
        query_service.initialize_services()
        query_service.connect_to_pinecone()
        
        logger.info("🎉 Phase 2 API ready to serve queries!")
        
    except Exception as e:
        logger.error(f"❌ STARTUP ERROR: {e}")
        logger.error(traceback.format_exc())
        initialization_status["error"] = str(e)
        # Don't raise - let health checks show the error

# Main API endpoint
@app.post("/hackrx/run", response_model=QueryResponse)
async def run_endpoint(req: QueryRequest, verified: bool = Depends(verify_token)):
    """Process insurance policy questions"""
    
    if not initialization_status["ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready. Status: {initialization_status}"
        )
    
    answers = []
    
    for question in req.questions:
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Search for relevant chunks
            relevant_chunks = query_service.search_relevant_chunks(question, top_k=8)
            
            # Generate answer  
            answer = query_service.generate_answer(question, relevant_chunks)
            answers.append(answer)
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            answers.append(f"Error processing question: {str(e)}")
    
    return QueryResponse(answers=answers)

# Health check endpoints
@app.get("/health")
async def health_check():
    """Quick health check"""
    return {
        "status": "healthy" if initialization_status["ready"] else "initializing",
        "message": "Phase 2 - Insurance Policy Query API",
        "initialization_status": initialization_status,
        "phase": 2,
        "description": "Query and retrieval only - no document processing"
    }

@app.get("/status")
async def detailed_status():
    """Detailed status information"""
    try:
        status_info = {
            "phase": 2,
            "services": initialization_status,
            "ready": initialization_status["ready"]
        }
        
        if query_service.index:
            try:
                stats = query_service.index.describe_index_stats()
                status_info["vector_stats"] = {
                    "total_vectors": stats.total_vector_count,
                    "index_fullness": stats.index_fullness
                }
            except Exception as stats_error:
                status_info["vector_stats"] = {"error": str(stats_error)}
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting detailed status: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/info")
async def get_info():
    """API information"""
    try:
        if not query_service.index:
            return {"status": "error", "message": "Index not connected"}
            
        stats = query_service.index.describe_index_stats()
        return {
            "phase": 2,
            "status": "ready",
            "total_vectors": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "embedding_model": "gemini-enhanced-semantic-embeddings",
            "llm_model": "gemini-1.5-flash",
            "functionality": "query_and_retrieval_only",
            "description": "Lightweight API for querying pre-indexed policy documents"
        }
    except Exception as e:
        logger.error(f"Error getting info: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    """Root endpoint information"""
    return {
        "message": "Insurance Policy RAG Query API - Phase 2",
        "version": "2.0.0",
        "phase": 2,
        "description": "Query and retrieval only - documents are pre-indexed in Phase 1",
        "endpoints": {
            "health": "/health",
            "status": "/status", 
            "info": "/info",
            "query": "/hackrx/run (POST)"
        },
        "note": "Ensure Phase 1 has been run to populate the Pinecone index"
    }

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
