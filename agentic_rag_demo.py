"""
🤖 AGENTIC RAG DEMO - COMPREHENSIVE EDUCATIONAL VERSION 
======================================================

This file demonstrates the complete implementation of Agentic RAG vs Traditional RAG
with detailed comments for educational purposes.

Key Concepts Demonstrated:
1. Traditional RAG: Fixed routing to local documents only
2. Agentic RAG: Intelligent routing between LOCAL, WEB, and HYBRID sources
3. Vector databases and semantic search
4. LLM-based routing decisions
5. Source attribution and transparency

Tech Stack:
- LangChain: RAG framework and document processing
- FAISS: Vector database for similarity search
- Streamlit: Web interface
- Groq: Fast LLM inference (Llama 3.1)
- HuggingFace: Text embeddings
- Serper: Web search API

Author: Educational Demo
Purpose: Teaching Agentic RAG concepts with hands-on implementation
"""

import streamlit as st
import time
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import os
import warnings

# =============================================================================
# 🔧 CONFIGURATION AND SETUP
# =============================================================================

# Suppress unnecessary warnings for cleaner demo
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure Streamlit page
st.set_page_config(
    page_title="Agentic RAG Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 🎨 CSS STYLING FOR PROFESSIONAL DEMO UI
# =============================================================================

st.markdown("""
<style>
    /* Main header with gradient effect */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    /* System-specific styling for visual differentiation */
    .traditional-rag {
        border-left: 5px solid #ff6b6b;
        background: linear-gradient(135deg, #ffe0e0 0%, #ffcccc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .agentic-rag {
        border-left: 5px solid #4ecdc4;
        background: linear-gradient(135deg, #e0f7f7 0%, #ccf2f2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Routing decision indicators */
    .local-route {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid #28a745;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .web-route {
        background: linear-gradient(135deg, #cce5ff 0%, #99d6ff 100%);
        color: #004085;
        border: 2px solid #007bff;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .hybrid-route {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        color: #00695c;
        border: 2px solid #00acc1;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🔑 API CONFIGURATION (Hardcoded for demo simplicity)
# =============================================================================

# API Keys - Check API_KEYS_LOCAL.md for actual keys
# Set these as environment variables or replace with your actual keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "your-serper-api-key-here")

# =============================================================================
# 🧠 CORE AI COMPONENTS INITIALIZATION
# =============================================================================

@st.cache_resource
def initialize_ai_components():
    """
    🔄 Initialize core AI components with caching for performance
    
    This function sets up:
    1. LLM (Large Language Model) - Groq's Llama 3.1 for fast inference
    2. Embeddings Model - HuggingFace model for semantic search
    3. Vector Database - FAISS for similarity search
    
    Returns:
        tuple: (llm, embeddings, vector_db) or (None, None, None) if failed
    """
    try:
        # Import required libraries
        from langchain_groq import ChatGroq
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document
        
        st.info("🔄 Initializing AI Components...")
        
        # 1. Initialize Large Language Model (Groq - Fast Inference)
        llm = ChatGroq(
            model='llama-3.1-8b-instant',  # Fast, efficient model
            temperature=0,                  # Deterministic responses
            max_tokens=500,                # Response length limit
            api_key=GROQ_API_KEY
        )
        
        # 2. Initialize Embeddings Model (HuggingFace - Semantic Understanding)
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',  # High-quality embeddings
            model_kwargs={'device': 'cpu'},                        # CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}           # Normalize for cosine similarity
        )
        
        # 3. Create sample documents for demonstration
        sample_documents = [
            Document(
                page_content=\"\"\"
                AGENTIC RAG: INTELLIGENT RETRIEVAL-AUGMENTED GENERATION
                
                Agentic RAG represents a revolutionary advancement in AI systems that combines
                the power of retrieval-augmented generation with intelligent agent capabilities.
                
                KEY FEATURES:
                • Intelligent Routing: Dynamically chooses between local documents and web search
                • Multi-Source Integration: Combines multiple information sources seamlessly
                • Quality Assessment: Evaluates and filters information for relevance
                • Adaptive Behavior: Learns and adapts to different types of queries
                
                COMPONENTS:
                1. Smart Router: Analyzes queries and determines optimal information sources
                2. Local Knowledge Base: Curated documents with high-quality information
                3. Web Search Agent: Real-time access to current information
                4. Context Synthesizer: Combines information from multiple sources
                5. Quality Evaluator: Ensures response accuracy and relevance
                
                ADVANTAGES OVER TRADITIONAL RAG:
                - Higher accuracy through intelligent source selection
                - Access to both curated and current information
                - Better handling of diverse query types
                - Improved user experience with faster, more relevant responses
                \"\"\",
                metadata={'source': 'agentic_rag_guide.pdf', 'page': 1, 'type': 'technical'}
            ),
            
            Document(
                page_content=\"\"\"
                MACHINE LEARNING FUNDAMENTALS FOR RAG SYSTEMS
                
                Machine Learning forms the foundation of modern RAG systems, providing
                the intelligence needed for document understanding and query processing.
                
                CORE ML CONCEPTS IN RAG:
                
                1. EMBEDDINGS AND VECTOR REPRESENTATIONS:
                   • Text Embeddings: Convert text into numerical vectors
                   • Semantic Similarity: Mathematical comparison of meaning
                   • Dimensionality: Typically 384-1536 dimensions for text
                   • Distance Metrics: Cosine similarity, Euclidean distance
                
                2. NEURAL NETWORKS IN RAG:
                   • Transformer Architecture: Foundation of modern LLMs
                   • Attention Mechanisms: Focus on relevant parts of text
                   • Pre-training: Large-scale language understanding
                   • Fine-tuning: Adaptation to specific tasks
                
                3. RETRIEVAL MECHANISMS:
                   • Dense Retrieval: Vector-based similarity search
                   • Sparse Retrieval: Traditional keyword matching
                   • Hybrid Approaches: Combining both methods
                   • Re-ranking: Improving initial retrieval results
                
                4. GENERATION TECHNIQUES:
                   • Conditional Generation: Using retrieved context
                   • Prompt Engineering: Optimizing model instructions
                   • Response Synthesis: Combining multiple sources
                   • Quality Control: Ensuring response accuracy
                \"\"\",
                metadata={'source': 'ml_fundamentals.pdf', 'page': 1, 'type': 'educational'}
            ),
            
            Document(
                page_content=\"\"\"
                VECTOR DATABASES: THE BACKBONE OF SEMANTIC SEARCH
                
                Vector databases are specialized systems designed for storing, indexing,
                and querying high-dimensional vectors efficiently.
                
                TECHNICAL ARCHITECTURE:
                
                1. VECTOR STORAGE:
                   • High-dimensional arrays (typically 100-2000 dimensions)
                   • Efficient memory management for large-scale data
                   • Compression techniques to reduce storage overhead
                   • Metadata association for contextual information
                
                2. INDEXING ALGORITHMS:
                   • HNSW (Hierarchical Navigable Small World): Fast approximate search
                   • IVF (Inverted File): Clustering-based indexing
                   • Product Quantization: Memory-efficient vector compression
                   • LSH (Locality Sensitive Hashing): Probabilistic similarity search
                
                3. SIMILARITY SEARCH:
                   • Cosine Similarity: Measures angle between vectors
                   • Euclidean Distance: Geometric distance in vector space
                   • Dot Product: Direct vector multiplication
                   • Manhattan Distance: Sum of absolute differences
                
                4. POPULAR VECTOR DATABASE SOLUTIONS:
                   • FAISS: Facebook's similarity search library
                   • Pinecone: Managed vector database service
                   • Weaviate: Open-source vector search engine
                   • Chroma: AI-native embedding database
                   • Qdrant: Vector similarity search engine
                
                PERFORMANCE CONSIDERATIONS:
                • Query latency: Sub-millisecond response times
                • Scalability: Millions to billions of vectors
                • Accuracy vs Speed: Trade-offs in approximate search
                • Memory efficiency: Optimizing for available resources
                \"\"\",
                metadata={'source': 'vector_databases.pdf', 'page': 1, 'type': 'technical'}
            )
        ]
        
        # 4. Create Vector Database with sample documents
        st.info("🔄 Creating vector database...")
        vector_db = FAISS.from_documents(sample_documents, embeddings)
        
        st.success("✅ AI Components initialized successfully!")
        return llm, embeddings, vector_db
        
    except Exception as e:
        st.error(f"❌ Failed to initialize AI components: {e}")
        return None, None, None

# =============================================================================
# 🔍 RETRIEVAL FUNCTIONS - CORE RAG FUNCTIONALITY
# =============================================================================

def get_local_content_with_scores(vector_db, query: str, k: int = 3) -> dict:
    """
    🎯 Retrieve relevant content from vector database with detailed metadata
    
    This is the core retrieval function that:
    1. Converts query to vector representation
    2. Performs similarity search against stored documents
    3. Returns relevant chunks with similarity scores
    4. Provides source attribution for transparency
    
    Args:
        vector_db: FAISS vector database instance
        query (str): User's question or search query
        k (int): Number of similar documents to retrieve
    
    Returns:
        dict: Contains content, source details, and similarity metrics
    """
    try:
        # Perform similarity search with scores (distance measurements)
        docs_with_scores = vector_db.similarity_search_with_score(query, k=k)
        
        content_pieces = []
        source_details = []
        
        # Process each retrieved document
        for i, (doc, distance_score) in enumerate(docs_with_scores):
            # Convert distance to similarity (higher = more similar)
            similarity_score = round(1 - distance_score, 3)
            
            # Extract document content and metadata
            content_pieces.append(doc.page_content)
            source_details.append({
                "chunk_id": i + 1,
                "similarity_score": similarity_score,
                "source_file": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "document_type": doc.metadata.get('type', 'Unknown'),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "content_length": len(doc.page_content)
            })
        
        # Calculate average similarity for quality assessment
        avg_similarity = round(sum([detail["similarity_score"] for detail in source_details]) / len(source_details), 3) if source_details else 0
        
        return {
            "content": ' '.join(content_pieces),
            "source_details": source_details,
            "total_chunks": len(docs_with_scores),
            "average_similarity": avg_similarity,
            "retrieval_quality": "High" if avg_similarity > 0.7 else "Medium" if avg_similarity > 0.5 else "Low"
        }
        
    except Exception as e:
        st.error(f"Error in local content retrieval: {e}")
        return {
            "content": "",
            "source_details": [],
            "total_chunks": 0,
            "average_similarity": 0,
            "retrieval_quality": "Failed"
        }

def get_web_content_with_metadata(query: str) -> dict:
    """
    🌐 Retrieve current information from web search with full metadata tracking
    
    This function provides access to real-time information by:
    1. Sending search query to Serper.dev API
    2. Processing search results and extracting relevant information
    3. Providing source attribution with links
    4. Tracking search metadata for transparency
    
    Args:
        query (str): Search query for web search
    
    Returns:
        dict: Contains search results, source metadata, and search statistics
    """
    url = 'https://google.serper.dev/search'
    payload = {'q': query, 'num': 5}  # Get top 5 results
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    
    search_start_time = time.time()
    
    try:
        # Execute web search
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
        search_duration = round(time.time() - search_start_time, 3)
        
        if response.status_code == 200:
            results = response.json()
            
            # Extract search metadata for transparency
            search_metadata = {
                "search_query": query,
                "search_duration": search_duration,
                "total_results": results.get('searchInformation', {}).get('totalResults', 'Unknown'),
                "search_engine_time": results.get('searchInformation', {}).get('searchTime', 'Unknown'),
                "status": "Success"
            }
            
            # Process organic search results
            if 'organic' in results:
                content_pieces = []
                sources = []
                
                for i, result in enumerate(results['organic'][:4]):  # Top 4 results
                    title = result.get('title', 'Untitled')
                    snippet = result.get('snippet', 'No description available')
                    link = result.get('link', '')
                    position = result.get('position', i+1)
                    
                    # Format content for LLM processing
                    content_pieces.append(f"**Source {i+1}: {title}**\\n{snippet}")
                    
                    # Store source metadata for attribution
                    sources.append({
                        "position": position,
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "domain": link.split('/')[2] if '//' in link else 'Unknown domain',
                        "snippet_length": len(snippet),
                        "relevance_rank": i + 1
                    })
                
                return {
                    "content": '\\n\\n'.join(content_pieces),
                    "sources": sources,
                    "search_metadata": search_metadata,
                    "success": True,
                    "result_count": len(sources),
                    "content_quality": "High" if len(sources) >= 3 else "Medium"
                }
        
        # Handle search failures
        return {
            "content": f"Limited web search results for: {query}",
            "sources": [],
            "search_metadata": {
                "search_query": query,
                "search_duration": search_duration,
                "error": f"HTTP {response.status_code}",
                "status": "Failed"
            },
            "success": False,
            "result_count": 0,
            "content_quality": "Poor"
        }
        
    except Exception as e:
        return {
            "content": f"Web search unavailable for: {query}",
            "sources": [],
            "search_metadata": {
                "search_query": query,
                "search_duration": round(time.time() - search_start_time, 3),
                "error": str(e),
                "status": "Error"
            },
            "success": False,
            "result_count": 0,
            "content_quality": "Failed"
        }

# =============================================================================
# 🧭 INTELLIGENT ROUTING - THE HEART OF AGENTIC RAG
# =============================================================================

def intelligent_query_router(llm, query: str, local_context: str) -> dict:
    """
    🎯 Advanced query router that makes intelligent decisions about information sources
    
    This is the core intelligence of Agentic RAG. It analyzes:
    1. Query content and intent
    2. Available local context relevance
    3. Temporal requirements (current vs historical info)
    4. Completeness of local information
    
    Routing Options:
    - LOCAL: Use only local documents (fast, curated)
    - WEB: Use only web search (current, comprehensive)
    - HYBRID: Use both sources (best of both worlds)
    
    Args:
        llm: Language model for decision making
        query (str): User's question
        local_context (str): Available local document content
    
    Returns:
        dict: Routing decision with detailed reasoning and confidence scores
    """
    
    # Construct detailed prompt for routing decision
    router_prompt = f\"\"\"
    🤖 AGENTIC RAG INTELLIGENT ROUTER
    ================================
    
    You are an advanced query router for an Agentic RAG system. Your job is to analyze
    the user's query and determine the optimal information source(s) to provide the
    best possible answer.
    
    USER QUERY: "{query}"
    
    AVAILABLE LOCAL CONTEXT (first 800 chars):
    {local_context[:800]}...
    
    ROUTING ANALYSIS FRAMEWORK:
    
    1. 🎯 CONTENT PRECISION ANALYSIS:
       - Does the local context contain the EXACT information needed?
       - Is the information complete and comprehensive?
       - Are there any gaps that need external information?
    
    2. ⏰ TEMPORAL REQUIREMENTS:
       - Does the query need current/real-time information?
       - Is this about recent events, prices, news, or trends?
       - Would historical information be sufficient?
    
    3. 📊 INFORMATION COMPLETENESS:
       - Can local context provide a complete answer?
       - Would additional sources enhance the response?
       - Is the topic partially covered but needs more detail?
    
    ROUTING DECISION RULES:
    
    • LOCAL: Choose when local context contains exact, complete information
             and query doesn't need current data
    
    • WEB: Choose when local context lacks information OR query needs
           current/real-time data (news, weather, prices, recent events)
    
    • HYBRID: Choose when local context provides good foundation but
              query would benefit from additional current information
    
    BE CONSERVATIVE: If uncertain about local content completeness, prefer WEB or HYBRID.
    
    REQUIRED OUTPUT FORMAT:
    Route: [LOCAL/WEB/HYBRID]
    Confidence: [HIGH/MEDIUM/LOW]
    Reasoning: [Detailed explanation of decision]
    Context_Match: [0.0-1.0 numerical score]
    Temporal_Need: [YES/NO]
    
    DECISION:
    \"\"\"
    
    try:
        # Get routing decision from LLM
        response = llm.invoke(router_prompt)
        decision_text = response.content.strip()
        
        # Parse structured response
        route = "LOCAL"  # Default fallback
        confidence = "MEDIUM"
        reasoning = decision_text
        context_match = 0.5
        temporal_need = "NO"
        
        # Extract structured information from response
        lines = decision_text.split('\\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Route:"):
                route_text = line.split(":", 1)[1].strip().upper()
                if "HYBRID" in route_text:
                    route = "HYBRID"
                elif "WEB" in route_text:
                    route = "WEB"
                else:
                    route = "LOCAL"
            elif line.startswith("Confidence:"):
                confidence = line.split(":", 1)[1].strip().upper()
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("Context_Match:"):
                try:
                    context_match = float(line.split(":", 1)[1].strip())
                except:
                    context_match = 0.5
            elif line.startswith("Temporal_Need:"):
                temporal_need = line.split(":", 1)[1].strip().upper()
        
        return {
            "route": route,
            "confidence": confidence.lower(),
            "reasoning": reasoning,
            "context_match_score": context_match,
            "temporal_requirement": temporal_need,
            "full_analysis": decision_text,
            "router_version": "Enhanced v2.0"
        }
        
    except Exception as e:
        # Fallback decision with error handling
        return {
            "route": "LOCAL",
            "confidence": "low",
            "reasoning": f"Router error: {str(e)} - defaulting to local search",
            "context_match_score": 0.0,
            "temporal_requirement": "UNKNOWN",
            "full_analysis": f"Error in routing analysis: {str(e)}",
            "router_version": "Fallback"
        }

# =============================================================================
# 🎯 RAG SYSTEM IMPLEMENTATIONS
# =============================================================================

def traditional_rag_query(llm, vector_db, query: str) -> dict:
    """
    📚 TRADITIONAL RAG: Simple, fixed approach
    
    Traditional RAG systems follow a basic pattern:
    1. Always search local documents
    2. Use simple similarity matching
    3. Generate answer from retrieved content
    4. No intelligence or adaptation
    
    Characteristics:
    - Fixed behavior (always local)
    - Basic retrieval (fewer documents)
    - Simple processing
    - Predictable but limited
    
    Args:
        llm: Language model for answer generation
        vector_db: Vector database for document retrieval
        query (str): User's question
    
    Returns:
        dict: Answer with basic metrics and limited source info
    """
    start_time = time.time()
    
    # STEP 1: Fixed local document retrieval (no intelligence)
    try:
        # Simple similarity search - fewer documents, basic approach
        docs = vector_db.similarity_search(query, k=2)  # Only 2 documents
        
        # Basic content processing - truncated for simplicity
        context_parts = []
        for doc in docs:
            # Truncate content to simulate basic processing
            truncated = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            context_parts.append(truncated)
        
        context = ' '.join(context_parts)
        
    except Exception as e:
        context = "Error retrieving local content"
    
    # STEP 2: Basic answer generation
    answer_prompt = f\"\"\"
    Answer the following question using only the provided context.
    Keep the response concise and direct.
    
    Context: {context}
    Question: {query}
    
    Answer:
    \"\"\"
    
    try:
        response = llm.invoke(answer_prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
    
    processing_time = time.time() - start_time
    
    return {
        "answer": answer,
        "source_type": "local",
        "processing_time": processing_time,
        "context_length": len(context),
        "route_decision": "LOCAL (Fixed)",
        "routing_explanation": "Traditional RAG always uses local documents with basic retrieval",
        "processing_steps": [
            "Fixed local document search",
            "Basic similarity matching",
            "Simple answer generation"
        ],
        "intelligence_level": "Basic",
        "documents_used": 2,
        "system_type": "Traditional RAG"
    }

def agentic_rag_query(llm, vector_db, query: str) -> dict:
    """
    🤖 AGENTIC RAG: Intelligent, adaptive approach
    
    Agentic RAG systems demonstrate advanced capabilities:
    1. Intelligent routing decisions
    2. Multi-source information integration
    3. Quality assessment and fallback mechanisms
    4. Comprehensive source attribution
    5. Adaptive behavior based on query type
    
    Key Features:
    - Smart routing (LOCAL/WEB/HYBRID)
    - Advanced retrieval with scoring
    - Multi-source synthesis
    - Quality checks and fallbacks
    - Full transparency and attribution
    
    Args:
        llm: Language model for routing and generation
        vector_db: Vector database for local retrieval
        query (str): User's question
    
    Returns:
        dict: Comprehensive response with full transparency and metrics
    """
    start_time = time.time()
    processing_steps = []
    
    # STEP 1: Intelligent Query Analysis
    processing_steps.append("🔍 Analyzing query intent and requirements")
    
    # Get local context sample for routing decision
    local_sample = get_local_content_with_scores(vector_db, query, k=3)
    sample_context = local_sample["content"]
    
    # STEP 2: Intelligent Routing Decision
    processing_steps.append("🧭 Making intelligent routing decision")
    routing_result = intelligent_query_router(llm, query, sample_context)
    route = routing_result["route"]
    
    # Initialize variables for different routing paths
    sources = []
    local_source_details = []
    web_metadata = {}
    context_parts = []
    
    # STEP 3: Execute Routing Decision
    if route == "LOCAL":
        processing_steps.append("📚 Retrieving from curated knowledge base")
        local_result = get_local_content_with_scores(vector_db, query, k=4)
        context = local_result["content"]
        source_type = "local"
        local_source_details = local_result["source_details"]
        
    elif route == "WEB":
        processing_steps.append("🌐 Searching web for current information")
        web_result = get_web_content_with_metadata(query)
        context = web_result["content"]
        sources = web_result["sources"]
        web_metadata = web_result.get("search_metadata", {})
        source_type = "web"
        
    else:  # HYBRID routing
        processing_steps.append("🔄 Retrieving from local knowledge base")
        local_result = get_local_content_with_scores(vector_db, query, k=3)
        local_context = local_result["content"]
        local_source_details = local_result["source_details"]
        context_parts.append(f"**📚 Local Knowledge:**\\n{local_context}")
        
        processing_steps.append("🌐 Searching web for additional information")
        web_result = get_web_content_with_metadata(query)
        web_context = web_result["content"]
        sources = web_result["sources"]
        web_metadata = web_result.get("search_metadata", {})
        context_parts.append(f"**🌐 Current Web Information:**\\n{web_context}")
        
        # Combine contexts for hybrid approach
        context = "\\n\\n".join(context_parts)
        source_type = "hybrid"
    
    # STEP 4: Advanced Answer Generation
    processing_steps.append("✨ Generating comprehensive response")
    
    if source_type == "hybrid":
        answer_prompt = f\"\"\"
        You are an expert AI assistant with access to both curated knowledge and current information.
        
        CONTEXT (combines local knowledge and current web information):
        {context}
        
        USER QUESTION: {query}
        
        INSTRUCTIONS:
        - Synthesize information from both local knowledge and web sources
        - Provide a comprehensive, well-structured answer
        - Distinguish between established knowledge and current information when relevant
        - Use clear formatting and be specific
        - Acknowledge sources when appropriate
        
        ANSWER:
        \"\"\"
    else:
        answer_prompt = f\"\"\"
        You are an expert AI assistant. Provide a comprehensive answer using the given context.
        
        CONTEXT: {context}
        
        USER QUESTION: {query}
        
        INSTRUCTIONS:
        - Provide a detailed, well-structured answer
        - Use bullet points or numbered lists when appropriate
        - Be specific and informative
        - Maintain a professional tone
        
        ANSWER:
        \"\"\"
    
    try:
        response = llm.invoke(answer_prompt)
        answer = response.content.strip()
        
        # Add source attribution for web/hybrid results
        if source_type in ["web", "hybrid"] and sources:
            source_info = "\\n\\n**🔗 Web Sources:**\\n"
            for i, source in enumerate(sources[:3], 1):
                source_info += f"{i}. {source.get('title', 'Unknown source')}\\n"
            answer += source_info
            
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"
    
    # STEP 5: Quality Check and Fallback (for LOCAL routing)
    if route == "LOCAL" and routing_result.get("confidence", "medium").lower() != "high":
        # Check for insufficient answers
        if len(answer) < 100 or any(phrase in answer.lower() for phrase in ["i don't have", "not available", "cannot find"]):
            processing_steps.append("⚠️ Local answer insufficient - executing fallback to web search")
            
            # Fallback to web search
            web_result = get_web_content_with_metadata(query)
            web_context = web_result["content"]
            web_sources = web_result["sources"]
            web_metadata = web_result.get("search_metadata", {})
            
            # Generate new answer with web content
            fallback_prompt = f\"\"\"
            The local knowledge was insufficient. Use this web information to answer the question.
            
            CONTEXT: {web_context}
            QUESTION: {query}
            
            Provide a comprehensive answer based on the current information available.
            
            ANSWER:
            \"\"\"
            
            try:
                response = llm.invoke(fallback_prompt)
                answer = response.content.strip()
                
                # Update result metadata to reflect fallback
                context = web_context
                source_type = "web"
                sources = web_sources
                route = "WEB (Fallback)"
                routing_result["reasoning"] += " [System detected insufficient local information and automatically switched to web search]"
                
            except Exception as e:
                answer = f"Fallback failed: {str(e)}"
    
    processing_time = time.time() - start_time
    
    # Compile comprehensive results
    return {
        "answer": answer,
        "source_type": source_type,
        "processing_time": processing_time,
        "context_length": len(context),
        "route_decision": route,
        "routing_explanation": routing_result["reasoning"],
        "routing_confidence": routing_result["confidence"],
        "context_match_score": routing_result.get("context_match_score", 0),
        "temporal_requirement": routing_result.get("temporal_requirement", "NO"),
        "full_routing_analysis": routing_result.get("full_analysis", ""),
        "processing_steps": processing_steps,
        "intelligence_level": "Advanced",
        "sources": sources,
        "local_source_details": local_source_details,
        "web_metadata": web_metadata,
        "total_sources_used": len(local_source_details) if source_type == "local" else len(sources),
        "system_type": "Agentic RAG",
        "router_version": routing_result.get("router_version", "Unknown")
    }

# =============================================================================
# 📊 VISUALIZATION AND UI COMPONENTS
# =============================================================================

def create_pipeline_visualization(system_type: str, route_decision: str = None):
    """
    📈 Create visual pipeline diagrams showing system flow
    
    Args:
        system_type (str): "traditional" or "agentic"
        route_decision (str): For agentic RAG - "LOCAL", "WEB", or "HYBRID"
    
    Returns:
        plotly.graph_objects.Figure: Interactive pipeline diagram
    """
    fig = go.Figure()
    
    if system_type == "traditional":
        # Traditional RAG - Simple linear flow
        fig.add_trace(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["User Query", "Fixed Router", "Local Documents", "Basic LLM", "Answer"],
                color=["#ffcccc", "#ff9999", "#ff6666", "#ff3333", "#ff0000"]
            ),
            link=dict(
                source=[0, 1, 2, 3],
                target=[1, 2, 3, 4],
                value=[1, 1, 1, 1],
                color=["#ff6666"] * 4
            )
        ))
        title = "📚 Traditional RAG Pipeline - Fixed & Basic"
        
    else:
        # Agentic RAG - Intelligent branching flow
        if route_decision == "LOCAL":
            colors = ["#ccffcc", "#99ff99", "#66ff66", "#33ff33", "#00ff00"]
            link_colors = ["#66ff66"] * 4
            source_label = "Local Documents"
        elif route_decision == "WEB":
            colors = ["#cce5ff", "#99d6ff", "#66c7ff", "#33b8ff", "#00a9ff"]
            link_colors = ["#66c7ff"] * 4
            source_label = "Web Search"
        else:  # HYBRID
            colors = ["#e0f7fa", "#b2ebf2", "#80deea", "#4dd0e1", "#26c6da"]
            link_colors = ["#80deea"] * 4
            source_label = "Local + Web Sources"
            
        fig.add_trace(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["User Query", "Smart Router", source_label, "Advanced LLM", "Enhanced Answer"],
                color=colors
            ),
            link=dict(
                source=[0, 1, 2, 3],
                target=[1, 2, 3, 4],
                value=[1, 1, 1, 1],
                color=link_colors
            )
        ))
        title = f"🤖 Agentic RAG Pipeline - {route_decision} Route"
    
    fig.update_layout(
        title_text=title,
        font_size=12,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# =============================================================================
# 🎮 MAIN STREAMLIT APPLICATION
# =============================================================================

def main():
    """
    🚀 Main Streamlit application - Educational Agentic RAG Demo
    
    This function creates the complete web interface demonstrating:
    1. Traditional RAG vs Agentic RAG comparison
    2. Interactive query testing
    3. Visual pipeline representations
    4. Detailed source attribution and transparency
    5. Performance metrics and analysis
    """
    
    # Application Header
    st.markdown('<div class="main-header">🤖 Agentic RAG Educational Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">'
        'Learn the Evolution from Traditional RAG to Intelligent Agentic RAG'
        '</div>', 
        unsafe_allow_html=True
    )
    
    # Initialize session state
    if 'components_loaded' not in st.session_state:
        st.session_state.components_loaded = False
        st.session_state.query_history = []
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 🎛️ Demo Controls")
        
        # Initialize AI components
        if not st.session_state.components_loaded:
            with st.spinner("🔄 Loading AI components..."):
                llm, embeddings, vector_db = initialize_ai_components()
                if llm and vector_db and embeddings:
                    st.session_state.llm = llm
                    st.session_state.vector_db = vector_db
                    st.session_state.embeddings = embeddings
                    st.session_state.components_loaded = True
                else:
                    st.error("❌ Failed to load AI components")
                    return
        
        st.success("✅ AI Components Ready!")
        
        # Sample questions for testing
        st.markdown("### 💬 Test Questions")
        
        st.markdown("**🏠 Local Knowledge Queries:**")
        local_questions = [
            "What is Agentic RAG and how does it work?",
            "Explain machine learning fundamentals",
            "How do vector databases enable semantic search?",
            "What are the key components of RAG systems?"
        ]
        
        for question in local_questions:
            if st.button(f"📚 {question}", key=f"local_{hash(question)}"):
                st.session_state.selected_query = question
        
        st.markdown("**🌐 Web Search Queries:**")
        web_questions = [
            "Latest AI news in 2024",
            "Current weather in San Francisco",
            "Recent developments in Large Language Models",
            "Today's technology trends"
        ]
        
        for question in web_questions:
            if st.button(f"🔍 {question}", key=f"web_{hash(question)}"):
                st.session_state.selected_query = question
        
        st.markdown("**🔄 Hybrid Queries (Local + Web):**")
        hybrid_questions = [
            "How is Agentic RAG being used in current AI applications?",
            "Latest developments in vector database technology?",
            "Current machine learning trends vs established RAG concepts?"
        ]
        
        for question in hybrid_questions:
            if st.button(f"🤝 {question}", key=f"hybrid_{hash(question)}"):
                st.session_state.selected_query = question
        
        # System architecture overview
        st.markdown("---")
        with st.expander("🏗️ System Architecture"):
            st.markdown("""
            **Traditional RAG:**
            - Fixed local document retrieval
            - No routing intelligence
            - Basic similarity matching
            - Limited adaptability
            
            **Agentic RAG:**
            - 🧭 Intelligent Router
            - 📚 Vector Database (FAISS)
            - 🌐 Web Search Agent (Serper)
            - 🤖 Advanced LLM (Groq Llama 3.1)
            - ⚖️ Quality Assessment
            - 🔄 Fallback Mechanisms
            """)
    
    # Main content area
    if not st.session_state.components_loaded:
        st.info("🔄 Please wait while AI components are loading...")
        return
    
    # Query input interface
    st.markdown("### 💬 Ask Your Question")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your question:",
            value=st.session_state.get('selected_query', ''),
            placeholder="Type your question here or select from sidebar examples..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        process_query = st.button("🚀 Compare Systems", type="primary")
    
    # Process query and show comparison
    if process_query and query:
        st.markdown("---")
        st.markdown("### 🔬 System Comparison Results")
        
        # Create two columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        # Traditional RAG Results
        with col1:
            st.markdown('<div class="traditional-rag">', unsafe_allow_html=True)
            st.markdown("#### 📚 Traditional RAG System")
            
            with st.spinner("Processing with Traditional RAG..."):
                traditional_result = traditional_rag_query(
                    st.session_state.llm,
                    st.session_state.vector_db,
                    query
                )
            
            # Display route decision
            st.markdown(
                f'<div class="local-route">{traditional_result["route_decision"]}</div>',
                unsafe_allow_html=True
            )
            st.info(traditional_result["routing_explanation"])
            
            # Processing steps
            st.markdown("**⚙️ Processing Steps:**")
            for i, step in enumerate(traditional_result["processing_steps"], 1):
                st.write(f"{i}. {step}")
            
            # Answer
            st.markdown("**💬 Answer:**")
            st.write(traditional_result["answer"])
            
            # Basic metrics
            st.markdown("**📊 System Metrics:**")
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("⏱️ Time", f"{traditional_result['processing_time']:.2f}s")
                st.metric("📄 Documents", traditional_result["documents_used"])
            with col1b:
                st.metric("📝 Context", f"{traditional_result['context_length']} chars")
                st.metric("🧠 Intelligence", traditional_result["intelligence_level"])
            
            # Pipeline visualization
            trad_fig = create_pipeline_visualization("traditional")
            st.plotly_chart(trad_fig, use_container_width=True)
            
            # Basic transparency info
            with st.expander("📋 Basic System Info"):
                st.info("Traditional RAG provides limited transparency:")
                st.markdown("- Fixed routing to local documents only")
                st.markdown("- Basic similarity matching (2 documents max)")
                st.markdown("- Simple processing pipeline")
                st.markdown("- No source attribution details")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Agentic RAG Results
        with col2:
            st.markdown('<div class="agentic-rag">', unsafe_allow_html=True)
            st.markdown("#### 🤖 Agentic RAG System")
            
            with st.spinner("Processing with Agentic RAG..."):
                agentic_result = agentic_rag_query(
                    st.session_state.llm,
                    st.session_state.vector_db,
                    query
                )
            
            # Display route decision with appropriate styling
            route_class = {
                "LOCAL": "local-route",
                "WEB": "web-route", 
                "HYBRID": "hybrid-route"
            }.get(agentic_result["route_decision"].split()[0], "local-route")
            
            st.markdown(
                f'<div class="{route_class}">{agentic_result["route_decision"]} Route</div>',
                unsafe_allow_html=True
            )
            st.success(agentic_result["routing_explanation"])
            
            # Advanced processing steps
            st.markdown("**⚙️ Advanced Processing Pipeline:**")
            for i, step in enumerate(agentic_result["processing_steps"], 1):
                st.write(f"{i}. {step}")
            
            # Answer
            st.markdown("**💬 Enhanced Answer:**")
            st.write(agentic_result["answer"])
            
            # Advanced metrics
            st.markdown("**📊 Advanced Metrics:**")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("⏱️ Time", f"{agentic_result['processing_time']:.2f}s")
                st.metric("🎯 Confidence", agentic_result["routing_confidence"].title())
            with col2b:
                st.metric("📝 Context", f"{agentic_result['context_length']} chars")
                st.metric("🧠 Intelligence", agentic_result["intelligence_level"])
            
            # Pipeline visualization
            agentic_fig = create_pipeline_visualization("agentic", agentic_result["route_decision"])
            st.plotly_chart(agentic_fig, use_container_width=True)
            
            # ===== COMPREHENSIVE TRANSPARENCY SECTION =====
            st.markdown("---")
            st.markdown("**🔍 Complete Transparency & Source Attribution**")
            
            # Routing analysis details
            with st.expander("🧭 Detailed Routing Analysis", expanded=True):
                st.markdown(f"**🎯 Route Decision:** {agentic_result['route_decision']}")
                st.markdown(f"**🎗️ Confidence Level:** {agentic_result['routing_confidence'].title()}")
                st.markdown(f"**📊 Context Match Score:** {agentic_result['context_match_score']:.3f}")
                st.markdown(f"**⏰ Temporal Requirement:** {agentic_result['temporal_requirement']}")
                st.markdown(f"**🤔 Reasoning:** {agentic_result['routing_explanation']}")
                
                with st.expander("📋 Complete Router Analysis"):
                    st.text(agentic_result.get('full_routing_analysis', 'Analysis not available'))
            
            # Source attribution based on routing decision
            if agentic_result['route_decision'].startswith('LOCAL') or agentic_result['source_type'] == 'local':
                with st.expander("📚 Local Document Sources", expanded=True):
                    local_details = agentic_result.get('local_source_details', [])
                    if local_details:
                        st.markdown(f"**📊 Total Chunks Used:** {len(local_details)}")
                        
                        for detail in local_details:
                            st.markdown(f"**📄 Chunk {detail['chunk_id']}:**")
                            col_a, col_b, col_c = st.columns([2, 1, 1])
                            
                            with col_a:
                                st.markdown(f"**📁 Source:** {detail['source_file']}")
                                st.markdown(f"**📃 Page:** {detail.get('page', 'N/A')}")
                            with col_b:
                                st.metric("🎯 Similarity", f"{detail['similarity_score']:.3f}")
                            with col_c:
                                st.metric("📏 Length", f"{detail['content_length']} chars")
                            
                            with st.expander(f"👁️ Preview Chunk {detail['chunk_id']}"):
                                st.text(detail['content_preview'])
                            
                            st.markdown("---")
            
            if agentic_result['route_decision'].startswith('WEB') or agentic_result['source_type'] in ['web', 'hybrid']:
                with st.expander("🌐 Web Search Sources", expanded=True):
                    web_sources = agentic_result.get('sources', [])
                    web_metadata = agentic_result.get('web_metadata', {})
                    
                    # Search metadata
                    st.markdown("**🔍 Search Information:**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("⏱️ Search Time", f"{web_metadata.get('search_duration', 0):.3f}s")
                    with col_b:
                        st.metric("📊 Total Results", web_metadata.get('total_results', 'Unknown'))
                    with col_c:
                        st.metric("📄 Sources Used", len(web_sources))
                    
                    st.markdown(f"**🔎 Search Query:** {web_metadata.get('search_query', query)}")
                    
                    # Individual web sources
                    if web_sources:
                        st.markdown("**🌐 Source Details:**")
                        for i, source in enumerate(web_sources, 1):
                            st.markdown(f"**🔗 Source {i}: {source.get('title', 'Unknown')}**")
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.markdown(f"**🏠 Domain:** {source.get('domain', 'Unknown')}")
                                if source.get('link'):
                                    st.markdown(f"**🔗 URL:** [Visit Source]({source['link']})")
                            with col_b:
                                st.metric("📍 Position", source.get('position', i))
                                st.metric("📏 Snippet Length", f"{source.get('snippet_length', 0)} chars")
                            
                            with st.expander(f"👁️ Preview Source {i}"):
                                st.text(source.get('snippet', 'No preview available'))
                            
                            st.markdown("---")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ===== COMPREHENSIVE COMPARISON ANALYSIS =====
        st.markdown("---")
        st.markdown("### 📊 Detailed Performance Comparison")
        
        # Performance metrics comparison
        comparison_data = pd.DataFrame({
            'System': ['Traditional RAG', 'Agentic RAG'],
            'Processing Time (s)': [traditional_result['processing_time'], agentic_result['processing_time']],
            'Source Type': [traditional_result['source_type'].title(), agentic_result['source_type'].title()],
            'Context Length': [traditional_result['context_length'], agentic_result['context_length']],
            'Intelligence Level': [traditional_result['intelligence_level'], agentic_result['intelligence_level']]
        })
        
        # Performance charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_time = px.bar(
                comparison_data,
                x='System',
                y='Processing Time (s)',
                title='⏱️ Processing Time Comparison',
                color='System',
                color_discrete_map={
                    'Traditional RAG': '#ff6b6b',
                    'Agentic RAG': '#4ecdc4'
                }
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            fig_context = px.bar(
                comparison_data,
                x='System',
                y='Context Length',
                title='📝 Context Utilization',
                color='System',
                color_discrete_map={
                    'Traditional RAG': '#ff6b6b',
                    'Agentic RAG': '#4ecdc4'
                }
            )
            st.plotly_chart(fig_context, use_container_width=True)
        
        with col3:
            intelligence_scores = {'Basic': 1, 'Advanced': 3}
            comparison_data['Intelligence Score'] = comparison_data['Intelligence Level'].map(intelligence_scores)
            
            fig_intel = px.bar(
                comparison_data,
                x='System',
                y='Intelligence Score',
                title='🧠 Intelligence Level',
                color='System',
                color_discrete_map={
                    'Traditional RAG': '#ff6b6b',
                    'Agentic RAG': '#4ecdc4'
                }
            )
            st.plotly_chart(fig_intel, use_container_width=True)
        
        # Key insights comparison
        st.markdown("### 🎯 Key Insights & Transparency Comparison")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("""
            **📚 Traditional RAG Characteristics:**
            - 🔴 **Fixed Behavior**: Always searches local documents
            - 🔴 **Basic Retrieval**: Simple similarity matching (2 docs max)
            - 🔴 **Limited Context**: Truncated content processing
            - 🔴 **No Intelligence**: Cannot adapt to query types
            - 🔴 **Poor Transparency**: Limited source attribution
            - ✅ **Predictable**: Consistent behavior and timing
            """)
        
        with insight_col2:
            st.markdown("""
            **🤖 Agentic RAG Advantages:**
            - 🟢 **Intelligent Routing**: Chooses optimal information sources
            - 🟢 **Advanced Processing**: Sophisticated retrieval and analysis
            - 🟢 **Rich Context**: Uses comprehensive information (4+ docs)
            - 🟢 **Adaptive Behavior**: Handles diverse query types effectively
            - 🟢 **Full Transparency**: Complete source attribution
            - 🟢 **Quality Assurance**: Fallback mechanisms for better results
            """)
        
        # Enhanced transparency features summary
        st.markdown("---")
        st.markdown("### 🔍 Enhanced Transparency Features in Agentic RAG")
        
        transparency_col1, transparency_col2 = st.columns(2)
        
        with transparency_col1:
            st.markdown("""
            **🧭 Routing Intelligence:**
            - Detailed routing decision reasoning
            - Confidence scores (HIGH/MEDIUM/LOW)
            - Context match scoring (0.0-1.0)
            - Temporal requirement analysis
            - Complete router analysis logs
            """)
        
        with transparency_col2:
            st.markdown("""
            **📊 Source Attribution:**
            - Document chunks with similarity scores
            - Source file and page information
            - Content previews for verification
            - Web source rankings and metadata
            - Search query optimization details
            """)
        
        # Decision tree visualization
        st.markdown("---")
        st.markdown("### 🌳 Decision Process Visualization")
        
        decision_tree_data = {
            'Query': [query[:50] + "..." if len(query) > 50 else query],
            'Router Analysis': [f"Context Match: {agentic_result.get('context_match_score', 0):.3f}"],
            'Final Decision': [agentic_result['route_decision']],
            'Confidence': [agentic_result.get('routing_confidence', 'unknown').title()],
            'Sources Used': [f"{agentic_result.get('total_sources_used', 0)} sources"]
        }
        
        tree_df = pd.DataFrame(decision_tree_data)
        st.dataframe(tree_df, use_container_width=True)
        
        # Add to query history for session tracking
        st.session_state.query_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'query': query,
            'traditional_route': traditional_result['route_decision'],
            'agentic_route': agentic_result['route_decision'],
            'traditional_time': traditional_result['processing_time'],
            'agentic_time': agentic_result['processing_time'],
            'routing_confidence': agentic_result.get('routing_confidence', 'unknown')
        })
    
    # Query history and analytics
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### 📈 Session Analytics")
        
        history_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Session analytics
        if len(st.session_state.query_history) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                avg_traditional = history_df['traditional_time'].mean()
                avg_agentic = history_df['agentic_time'].mean()
                
                st.metric("📊 Avg Traditional RAG Time", f"{avg_traditional:.2f}s")
                st.metric("📊 Avg Agentic RAG Time", f"{avg_agentic:.2f}s")
            
            with col2:
                # Route distribution for agentic RAG
                route_counts = history_df['agentic_route'].value_counts()
                st.write("🎯 Agentic RAG Route Distribution:")
                for route, count in route_counts.items():
                    st.write(f"   • {route}: {count} queries")
        
        if st.button("🗑️ Clear Session History"):
            st.session_state.query_history = []
            st.rerun()

# =============================================================================
# 🚀 APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()