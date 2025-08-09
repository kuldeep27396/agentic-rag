"""
Enhanced Agentic RAG Demo Web Interface

Advanced Streamlit application with PDF upload, dynamic training, 
better Traditional vs Agentic RAG differentiation, and visual pipeline components.

Run with: streamlit run enhanced_streamlit_app.py
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
import io
import base64
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Page configuration
st.set_page_config(
    page_title="Enhanced Agentic RAG Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin: 1rem 0;
    }
    .pipeline-box {
        border: 2px solid #e1e5e9;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .traditional-rag {
        border-left: 5px solid #ff6b6b;
        background: linear-gradient(135deg, #ffe0e0 0%, #ffcccc 100%);
    }
    .agentic-rag {
        border-left: 5px solid #4ecdc4;
        background: linear-gradient(135deg, #e0f7f7 0%, #ccf2f2 100%);
    }
    .routing-decision {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.8rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .local-route {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 2px solid #28a745;
    }
    .web-route {
        background: linear-gradient(135deg, #cce5ff 0%, #99d6ff 100%);
        color: #004085;
        border: 2px solid #007bff;
    }
    .fixed-route {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .hybrid-route {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        color: #00695c;
        border: 2px solid #00acc1;
    }
    .upload-area {
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1rem 0;
    }
    .pipeline-step {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        border-radius: 20px;
        background: #e9ecef;
        border: 1px solid #ced4da;
        font-size: 0.9rem;
    }
    .step-active {
        background: #28a745;
        color: white;
        border: 1px solid #28a745;
    }
    .step-traditional {
        background: #ff6b6b;
        color: white;
        border: 1px solid #ff6b6b;
    }
    .step-agentic {
        background: #4ecdc4;
        color: white;
        border: 1px solid #4ecdc4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# API Keys - Will be set from Streamlit secrets or environment variables
def get_api_keys():
    """Get API keys from Streamlit secrets or environment variables"""
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        groq_key = st.secrets.get("GROQ_API_KEY", "")
        serper_key = st.secrets.get("SERPER_API_KEY", "")
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        
        # If found in secrets, return them
        if groq_key and serper_key:
            return groq_key, serper_key, gemini_key
    except:
        pass
    
    # Fallback to environment variables (for other deployments)
    groq_key = os.getenv("GROQ_API_KEY", "")
    serper_key = os.getenv("SERPER_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    
    return groq_key, serper_key, gemini_key

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.query_history = []
    st.session_state.components_loaded = False
    st.session_state.custom_docs_loaded = False
    st.session_state.dynamic_questions = []
    st.session_state.current_knowledge_base = "default"

@st.cache_resource
def initialize_base_components():
    """Initialize base RAG components"""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq
        from langchain.schema import Document
        
        # Get API keys from session state or environment
        groq_key, serper_key, gemini_key = get_api_keys()
        
        if not groq_key:
            st.error("❌ Groq API key required for LLM functionality")
            return None, None
        
        # Initialize LLM
        llm = ChatGroq(
            model='llama-3.1-8b-instant',
            temperature=0,
            max_tokens=500,
            timeout=None,
            max_retries=2,
            api_key=groq_key
        )
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-mpnet-base-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        return llm, embeddings
        
    except Exception as e:
        st.error(f"Error initializing base components: {e}")
        return None, None

def create_default_knowledge_base(embeddings):
    """Create default knowledge base with sample documents"""
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS
    
    default_docs = [
        Document(
            page_content="""Agentic RAG: The Future of Intelligent Information Retrieval
            
            Agentic RAG represents a revolutionary approach to information retrieval and generation,
            combining the decision-making capabilities of AI agents with retrieval-augmented generation.
            
            Unlike traditional RAG systems that blindly retrieve from a fixed knowledge base,
            Agentic RAG systems exhibit intelligence by:
            
            1. **Smart Routing**: Automatically deciding between local knowledge and web search
            2. **Quality Assessment**: Evaluating the relevance and quality of retrieved information
            3. **Adaptive Querying**: Rewriting queries for better retrieval results
            4. **Multi-Source Integration**: Seamlessly combining information from multiple sources
            5. **Context Awareness**: Understanding query intent and user needs
            
            Key advantages over traditional approaches:
            - Higher accuracy through intelligent source selection
            - Better performance by avoiding unnecessary operations
            - Enhanced user experience with contextually relevant responses
            - Improved reliability through quality checks and fallbacks
            - Future-proof architecture that adapts to changing information needs""",
            metadata={'source': 'agentic_rag_whitepaper.pdf', 'page': 1, 'type': 'technical'}
        ),
        Document(
            page_content="""Artificial Intelligence and Machine Learning: Core Concepts
            
            Machine learning forms the backbone of modern AI applications, enabling systems
            to learn and improve from experience without explicit programming.
            
            The three fundamental paradigms of machine learning are:
            
            **1. Supervised Learning**
            - Learns from labeled training examples
            - Goal: Predict outputs for new inputs
            - Examples: Classification, regression, object detection
            - Algorithms: Neural networks, decision trees, SVM, random forests
            - Applications: Email filtering, medical diagnosis, financial forecasting
            
            **2. Unsupervised Learning**
            - Discovers patterns in unlabeled data
            - Goal: Find hidden structures and relationships
            - Examples: Clustering, dimensionality reduction, anomaly detection
            - Algorithms: K-means, PCA, autoencoders, GANs
            - Applications: Customer segmentation, data compression, fraud detection
            
            **3. Reinforcement Learning**
            - Learns through interaction with environment
            - Goal: Maximize cumulative rewards through optimal actions
            - Examples: Game playing, robotics, resource allocation
            - Algorithms: Q-learning, policy gradients, actor-critic methods
            - Applications: Autonomous vehicles, trading systems, recommendation engines
            
            Modern applications leverage deep learning, transfer learning, and ensemble methods
            to achieve state-of-the-art performance across diverse domains.""",
            metadata={'source': 'ai_ml_handbook.pdf', 'page': 1, 'type': 'educational'}
        )
    ]
    
    return FAISS.from_documents(default_docs, embeddings)

def process_uploaded_pdf(uploaded_file, embeddings):
    """Process uploaded PDF and create vector database"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        from langchain_community.vectorstores import FAISS
        import tempfile
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        doc_chunks = text_splitter.split_documents(documents)
        
        # Create vector database
        vector_db = FAISS.from_documents(doc_chunks, embeddings)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vector_db, doc_chunks
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None, None

def generate_dynamic_questions(doc_chunks, llm):
    """Generate relevant questions based on uploaded content"""
    if not doc_chunks:
        return []
    
    # Sample content from chunks
    sample_content = " ".join([chunk.page_content[:200] for chunk in doc_chunks[:3]])
    
    question_prompt = f"""Based on the following document content, generate 6 relevant questions that would test both local knowledge retrieval and require current/external information.

Content: {sample_content}

Generate questions in two categories:
1. Three questions that can be answered from the document content (LOCAL)
2. Three questions that would require external/current information (WEB)

Format as:
LOCAL: question1
LOCAL: question2  
LOCAL: question3
WEB: question1
WEB: question2
WEB: question3

Questions:"""
    
    try:
        response = llm.invoke(question_prompt)
        questions_text = response.content.strip()
        
        local_questions = []
        web_questions = []
        
        for line in questions_text.split('\n'):
            line = line.strip()
            if line.startswith('LOCAL:'):
                local_questions.append(line[6:].strip())
            elif line.startswith('WEB:'):
                web_questions.append(line[4:].strip())
        
        return local_questions, web_questions
        
    except:
        # Fallback questions
        return [
            "What is the main topic of the document?",
            "What are the key concepts mentioned?",
            "What are the main benefits discussed?"
        ], [
            "What are the latest developments in this field?",
            "What are current market trends related to this topic?",
            "What are recent news articles about this subject?"
        ]

def get_local_content(vector_db, query: str, k: int = 3) -> dict:
    """Retrieve content from vector database with detailed information"""
    try:
        docs_with_scores = vector_db.similarity_search_with_score(query, k=k)
        
        content_pieces = []
        source_details = []
        
        for i, (doc, score) in enumerate(docs_with_scores):
            content_pieces.append(doc.page_content)
            source_details.append({
                "chunk_id": i + 1,
                "similarity_score": round(1 - score, 3),  # Convert distance to similarity
                "source_file": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "content_length": len(doc.page_content)
            })
        
        return {
            "content": ' '.join(content_pieces),
            "source_details": source_details,
            "total_chunks": len(docs_with_scores),
            "avg_similarity": round(sum([1 - score for _, score in docs_with_scores]) / len(docs_with_scores), 3) if docs_with_scores else 0
        }
    except Exception as e:
        print(f"Error in get_local_content: {e}")
        return {
            "content": "",
            "source_details": [],
            "total_chunks": 0,
            "avg_similarity": 0
        }

def traditional_rag_simple_retrieval(vector_db, query: str, k: int = 2) -> str:
    """Simplified retrieval for traditional RAG - more basic approach"""
    try:
        # Use fewer documents and simpler retrieval
        docs = vector_db.similarity_search(query, k=k)
        # Just concatenate without sophisticated processing
        content = " ".join([doc.page_content[:300] for doc in docs])  # Limit content
        return content
    except:
        return ""

def check_local_knowledge_enhanced(llm, query: str, context: str) -> dict:
    """Enhanced router with detailed reasoning and confidence scoring"""
    router_prompt = f"""You are an advanced query router for an Agentic RAG system. You must be VERY CAREFUL and PRECISE about routing decisions.

Query: "{query}"
Available Local Context: {context[:800]}...

CRITICAL ANALYSIS REQUIREMENTS:
1. EXACT Content Match: Does the local context contain the EXACT information needed to answer this specific query? Not just related topics, but the precise answer.
2. Completeness Check: Can you provide a COMPLETE answer using ONLY the local context?
3. Information Specificity: Is the query asking for specific details, personal information, or data that must be explicitly present?
4. Temporal Requirements: Does the query need current/real-time information?

STRICT Routing Decision Rules:
- LOCAL: ONLY if the context contains the EXACT answer to the query
- WEB: Current events, real-time data, personal information not in context, specific details absent from context
- HYBRID: Query needs both foundational knowledge (in context) AND current/additional information

Be CONSERVATIVE: If you're not 100% certain the local context can fully answer the query, route to WEB or HYBRID.

Provide your analysis in this format:
Route: LOCAL, WEB, or HYBRID
Confidence: HIGH/MEDIUM/LOW
Reasoning: [Explain WHY you chose this route - be specific about what information is/isn't available]
Context_Match: [0.0-1.0 score for how well context matches the SPECIFIC query]
Temporal_Need: [YES/NO - does query need current information]

Decision:"""
    
    try:
        response = llm.invoke(router_prompt)
        decision_text = response.content.strip()
        
        # Parse the structured response
        lines = decision_text.split('\n')
        route = "LOCAL"
        confidence = "MEDIUM"
        reasoning = decision_text
        context_match = 0.5
        temporal_need = "NO"
        
        for line in lines:
            line = line.strip()
            if line.startswith("Route:"):
                if "HYBRID" in line.upper():
                    route = "HYBRID"
                elif "WEB" in line.upper():
                    route = "WEB"
                else:
                    route = "LOCAL"
            elif line.startswith("Confidence:"):
                confidence = line.split(":")[1].strip()
            elif line.startswith("Reasoning:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("Context_Match:"):
                try:
                    context_match = float(line.split(":")[1].strip())
                except:
                    context_match = 0.5
            elif line.startswith("Temporal_Need:"):
                temporal_need = line.split(":")[1].strip()
        
        return {
            "route": route,
            "reasoning": reasoning,
            "confidence": confidence.lower(),
            "context_match_score": context_match,
            "temporal_requirement": temporal_need,
            "full_analysis": decision_text
        }
    except Exception as e:
        return {
            "route": "LOCAL",
            "reasoning": f"Router error: {str(e)} - defaulting to local",
            "confidence": "low",
            "context_match_score": 0.0,
            "temporal_requirement": "UNKNOWN",
            "full_analysis": "Error in routing analysis"
        }

def get_web_content_enhanced(query: str) -> dict:
    """Enhanced web content retrieval with detailed source information"""
    # Get API keys from session state
    groq_key, serper_key, gemini_key = get_api_keys()
    
    if not serper_key:
        return {
            "content": "Web search unavailable - Serper API key not configured",
            "sources": [],
            "search_metadata": {
                "search_query": query,
                "error": "No API key configured",
                "status": "Failed"
            },
            "success": False,
            "result_count": 0
        }
    
    url = 'https://google.serper.dev/search'
    payload = {'q': query, 'num': 5}
    headers = {
        'X-API-KEY': serper_key,
        'Content-Type': 'application/json'
    }
    
    search_start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
        search_time = time.time() - search_start_time
        
        if response.status_code == 200:
            results = response.json()
            
            # Extract search metadata
            search_metadata = {
                "search_query": query,
                "search_time": round(search_time, 3),
                "total_results": results.get('searchInformation', {}).get('totalResults', 'Unknown'),
                "search_time_google": results.get('searchInformation', {}).get('searchTime', 'Unknown')
            }
            
            if 'organic' in results:
                content_pieces = []
                sources = []
                
                for i, result in enumerate(results['organic'][:4]):
                    title = result.get('title', 'Untitled')
                    snippet = result.get('snippet', 'No description available')
                    link = result.get('link', '')
                    position = result.get('position', i+1)
                    
                    content_pieces.append(f"**Source {i+1}**: {title}\n{snippet}")
                    sources.append({
                        "position": position,
                        "title": title, 
                        "link": link,
                        "snippet": snippet,
                        "domain": link.split('/')[2] if '//' in link else 'Unknown domain',
                        "snippet_length": len(snippet)
                    })
                
                return {
                    "content": '\n\n'.join(content_pieces),
                    "sources": sources,
                    "search_metadata": search_metadata,
                    "success": True,
                    "result_count": len(sources),
                    "search_query_used": query
                }
        
        return {
            "content": f"Limited web search results for: {query}",
            "sources": [],
            "search_metadata": {
                "search_query": query,
                "search_time": round(search_time, 3),
                "error": f"HTTP {response.status_code}"
            },
            "success": False,
            "result_count": 0,
            "search_query_used": query
        }
        
    except Exception as e:
        return {
            "content": f"Web search error for query: {query}",
            "sources": [],
            "search_metadata": {
                "search_query": query,
                "search_time": round(time.time() - search_start_time, 3),
                "error": str(e)
            },
            "success": False,
            "result_count": 0,
            "search_query_used": query
        }

def generate_answer_enhanced(llm, context: str, query: str, source_type: str, sources: list = None) -> str:
    """Enhanced answer generation with source attribution"""
    if source_type == "hybrid":
        answer_prompt = f"""You are an expert AI assistant. Use the provided context from both local knowledge and current web information to answer the user's question comprehensively.

Context (includes both local knowledge and current web information): {context}

Question: {query}

Instructions:
- Synthesize information from both local knowledge and web sources
- Provide a comprehensive answer that leverages both types of information
- Clearly distinguish between established knowledge and current information when relevant
- Use clear formatting with bullet points or numbered lists when appropriate
- Be specific and informative
- Maintain a professional and helpful tone

Answer:"""
    else:
        answer_prompt = f"""You are an expert AI assistant. Use the provided context to answer the user's question comprehensively and accurately.

Context: {context}

Question: {query}

Instructions:
- Provide a detailed, well-structured answer based on the context
- Use clear formatting with bullet points or numbered lists when appropriate
- Be specific and informative
- If the context is limited, acknowledge this appropriately
- Maintain a professional and helpful tone

Answer:"""
    
    try:
        response = llm.invoke(answer_prompt)
        answer = response.content.strip()
        
        # Add source information
        if source_type in ["web", "hybrid"] and sources:
            source_info = "\n\n**Web Sources:**\n"
            for i, source in enumerate(sources[:3], 1):
                source_info += f"{i}. {source.get('title', 'Unknown source')}\n"
            answer += source_info
        
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def traditional_rag_query_enhanced(llm, vector_db, query: str) -> dict:
    """Enhanced traditional RAG with simpler, more basic behavior"""
    start_time = time.time()
    
    # Traditional RAG: Simple, fixed approach
    context = traditional_rag_simple_retrieval(vector_db, query, k=2)  # Fewer docs
    answer = generate_answer_enhanced(llm, context, query, "local")
    
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
        "intelligence_level": "Basic"
    }

def agentic_rag_query_enhanced(llm, vector_db, query: str) -> dict:
    """Enhanced Agentic RAG with sophisticated routing and detailed transparency"""
    start_time = time.time()
    processing_steps = []
    
    # Step 1: Intelligent analysis
    processing_steps.append("Analyzing query intent and context")
    local_sample = get_local_content(vector_db, "sample", k=2)
    sample_context = local_sample["content"] if isinstance(local_sample, dict) else str(local_sample)
    
    # Step 2: Advanced routing decision with detailed analysis
    processing_steps.append("Making intelligent routing decision with confidence scoring")
    # Get actual relevant context for the query instead of just sample
    query_context = get_local_content(vector_db, query, k=3)
    actual_context = query_context["content"] if isinstance(query_context, dict) else str(query_context)
    routing_result = check_local_knowledge_enhanced(llm, query, actual_context)
    route = routing_result["route"]
    
    # Step 3: Source-specific retrieval with detailed tracking
    sources = []
    local_source_details = []
    web_metadata = {}
    context_parts = []
    
    if route == "LOCAL":
        processing_steps.append("Retrieving from curated knowledge base with similarity scoring")
        local_result = get_local_content(vector_db, query, k=4)
        context = local_result["content"]
        source_type = "local"
        local_source_details = local_result["source_details"]
        
    elif route == "WEB":
        processing_steps.append("Searching web for current information with source tracking")
        web_result = get_web_content_enhanced(query)
        context = web_result["content"]
        sources = web_result["sources"]
        web_metadata = web_result.get("search_metadata", {})
        source_type = "web"
        
    else:  # HYBRID routing
        processing_steps.append("Retrieving from local knowledge base")
        local_result = get_local_content(vector_db, query, k=3)
        local_context = local_result["content"]
        local_source_details = local_result["source_details"]
        context_parts.append(f"**Local Knowledge:**\n{local_context}")
        
        processing_steps.append("Searching web for additional current information")
        web_result = get_web_content_enhanced(query)
        web_context = web_result["content"]
        sources = web_result["sources"]
        web_metadata = web_result.get("search_metadata", {})
        context_parts.append(f"**Current Web Information:**\n{web_context}")
        
        # Combine both contexts
        context = "\n\n".join(context_parts)
        source_type = "hybrid"
    
    # Step 4: Enhanced answer generation with quality check
    processing_steps.append("Generating contextually-aware response with source attribution")
    answer = generate_answer_enhanced(llm, context, query, source_type, sources)
    
    # Step 5: Quality check for LOCAL routing - fallback to WEB if answer is poor
    if route == "LOCAL" and routing_result.get("confidence", "medium").lower() != "high":
        # Check if the answer seems incomplete or generic
        if len(answer) < 100 or "I don't have" in answer or "not available" in answer.lower() or "cannot find" in answer.lower():
            processing_steps.append("Local answer insufficient - falling back to web search")
            web_result = get_web_content_enhanced(query)
            web_context = web_result["content"]
            web_sources = web_result["sources"]
            web_metadata = web_result.get("search_metadata", {})
            
            # Generate new answer with web content
            fallback_answer = generate_answer_enhanced(llm, web_context, query, "web", web_sources)
            
            # Update results to reflect the fallback
            answer = fallback_answer
            context = web_context
            source_type = "web"
            sources = web_sources
            web_metadata = web_metadata
            route = "WEB (Fallback)"
            routing_result["reasoning"] += " [System detected insufficient local information and switched to web search]"
    
    processing_time = time.time() - start_time
    
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
        "total_sources_used": len(local_source_details) if source_type == "local" else len(sources)
    }

def create_pipeline_visualization(system_type: str, route_decision: str = None, processing_steps: list = None):
    """Create enhanced pipeline visualization"""
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
                color=["#ff6666", "#ff6666", "#ff6666", "#ff6666"]
            )
        ))
        title = "Traditional RAG Pipeline - Fixed & Simple"
        
    else:
        # Agentic RAG - Intelligent branching flow
        if route_decision == "LOCAL":
            colors = ["#ccffcc", "#99ff99", "#66ff66", "#33ff33", "#00ff00"]
            link_colors = ["#66ff66", "#66ff66", "#66ff66", "#66ff66"]
            source_label = "Local Source"
        elif route_decision == "WEB":
            colors = ["#cce5ff", "#99d6ff", "#66c7ff", "#33b8ff", "#00a9ff"]
            link_colors = ["#66c7ff", "#66c7ff", "#66c7ff", "#66c7ff"]
            source_label = "Web Source"
        else:  # HYBRID
            colors = ["#e0f7fa", "#b2ebf2", "#80deea", "#4dd0e1", "#26c6da"]
            link_colors = ["#80deea", "#80deea", "#80deea", "#80deea"]
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
        title = f"Agentic RAG Pipeline - {route_decision} Route"
    
    fig.update_layout(
        title_text=title,
        font_size=12,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def create_processing_steps_visual(steps: list, system_type: str):
    """Create visual representation of processing steps"""
    step_html = "<div style='margin: 1rem 0;'>"
    
    for i, step in enumerate(steps, 1):
        if system_type == "traditional":
            step_class = "pipeline-step step-traditional"
        else:
            step_class = "pipeline-step step-agentic"
            
        step_html += f"<span class='{step_class}'>{i}. {step}</span>"
        if i < len(steps):
            step_html += " → "
    
    step_html += "</div>"
    return step_html

def main():
    """Enhanced main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🤖 Enhanced Agentic RAG Demo</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Advanced AI-Powered Information Retrieval with PDF Upload & Dynamic Training</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # API Keys Status Section
        st.markdown("### 🔑 API Configuration")
        
        # Check API key availability from environment
        groq_key, serper_key, gemini_key = get_api_keys()
        
        if groq_key and serper_key:
            st.success("✅ API keys configured from environment")
            st.markdown("**🟢 Status:** Ready to use all features")
        else:
            st.error("❌ API keys not found in environment variables")
            st.markdown("**🔴 Status:** Please configure environment variables")
            st.info("💡 Required: GROQ_API_KEY, SERPER_API_KEY, GEMINI_API_KEY")
        
        st.markdown("---")
        st.markdown("### 🎛️ Knowledge Base Setup")
        
        # PDF Upload Section
        st.markdown("#### 📄 Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files to create custom knowledge base",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to train the RAG system on your content"
        )
        
        if uploaded_files and st.button("🚀 Process & Train", type="primary"):
            with st.spinner("Processing uploaded documents..."):
                # Initialize base components if not already done
                if not st.session_state.components_loaded:
                    llm, embeddings = initialize_base_components()
                    if llm and embeddings:
                        st.session_state.llm = llm
                        st.session_state.embeddings = embeddings
                        st.session_state.components_loaded = True
                
                # Process uploaded PDFs
                all_chunks = []
                for uploaded_file in uploaded_files:
                    st.write(f"Processing: {uploaded_file.name}")
                    vector_db, chunks = process_uploaded_pdf(uploaded_file, st.session_state.embeddings)
                    if chunks:
                        all_chunks.extend(chunks)
                
                if all_chunks:
                    # Create combined vector database
                    from langchain_community.vectorstores import FAISS
                    st.session_state.vector_db = FAISS.from_documents(all_chunks, st.session_state.embeddings)
                    st.session_state.custom_docs_loaded = True
                    st.session_state.current_knowledge_base = "custom"
                    
                    # Generate dynamic questions
                    local_q, web_q = generate_dynamic_questions(all_chunks, st.session_state.llm)
                    st.session_state.dynamic_questions = {
                        'local': local_q,
                        'web': web_q
                    }
                    
                    st.success(f"✅ Processed {len(uploaded_files)} files ({len(all_chunks)} chunks)")
                    st.rerun()
        
        # Initialize default components if no custom docs
        if not st.session_state.components_loaded and st.session_state.api_keys_configured:
            with st.spinner("Loading default AI components..."):
                llm, embeddings = initialize_base_components()
                if llm and embeddings:
                    st.session_state.llm = llm
                    st.session_state.embeddings = embeddings
                    st.session_state.vector_db = create_default_knowledge_base(embeddings)
                    st.session_state.components_loaded = True
                    st.session_state.current_knowledge_base = "default"
        
        # Knowledge Base Status
        st.markdown("#### 📊 Current Knowledge Base")
        if st.session_state.get('current_knowledge_base') == "custom":
            st.success("🔄 Custom documents loaded")
        else:
            st.info("📚 Using default knowledge base")
        
        # Dynamic Questions Section
        st.markdown("#### 💬 Sample Questions")
        
        if st.session_state.get('dynamic_questions'):
            st.markdown("**🏠 Local Knowledge Queries:**")
            for question in st.session_state.dynamic_questions['local']:
                if st.button(f"📚 {question}", key=f"local_{hash(question)}"):
                    st.session_state.selected_query = question
            
            st.markdown("**🌐 Web Search Queries:**")
            for question in st.session_state.dynamic_questions['web']:
                if st.button(f"🔍 {question}", key=f"web_{hash(question)}"):
                    st.session_state.selected_query = question
        else:
            # Default questions
            default_local = [
                "What is Agentic RAG?",
                "Explain machine learning types",
                "How do AI systems work?",
                "What are the key concepts mentioned?"
            ]
            
            default_web = [
                "Latest AI news in 2024",
                "Current weather in San Francisco", 
                "Today's technology trends"
            ]
            
            default_hybrid = [
                "How is Agentic RAG used in current applications?",
                "Latest developments in vector databases?",
                "Current ML trends vs RAG concepts?"
            ]
            
            st.markdown("**🏠 Local Knowledge Queries:**")
            for question in default_local:
                if st.button(f"📚 {question}", key=f"def_local_{hash(question)}"):
                    st.session_state.selected_query = question
            
            st.markdown("**🌐 Web Search Queries:**")
            for question in default_web:
                if st.button(f"🔍 {question}", key=f"def_web_{hash(question)}"):
                    st.session_state.selected_query = question
                    
            st.markdown("**🔄 Hybrid Queries (Local + Web):**")
            for question in default_hybrid:
                if st.button(f"🤝 {question}", key=f"def_hybrid_{hash(question)}"):
                    st.session_state.selected_query = question
        
        st.markdown("---")
        
        # System Information
        with st.expander("🔧 System Architecture"):
            st.markdown("""
            **Traditional RAG:**
            - ❌ Fixed local-only retrieval
            - ❌ Basic similarity search
            - ❌ Simple answer generation
            - ❌ No intelligence or adaptation
            
            **Agentic RAG:**
            - ✅ Intelligent routing system
            - ✅ Advanced retrieval strategies
            - ✅ Multi-source information fusion
            - ✅ Context-aware processing
            - ✅ Quality assessment & optimization
            """)
    
    # Main content area
    if not st.session_state.api_keys_configured:
        st.warning("⚠️ Please configure your API keys in the sidebar to use the demo")
        st.info("💡 You can get free API keys from:")
        st.markdown("- 🤖 **Groq**: https://console.groq.com/ (Required)")
        st.markdown("- 🔍 **Serper**: https://serper.dev/ (Required)")
        st.markdown("- 🧠 **Gemini**: https://makersuite.google.com/ (Optional)")
        return
    
    if not st.session_state.components_loaded:
        st.info("🔄 Please wait while components are loading...")
        return
    
    # Query input section
    st.markdown('<div class="sub-header">💬 Ask Your Question</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your question:",
            value=st.session_state.get('selected_query', ''),
            placeholder="Ask anything - the system will intelligently route to the best source...",
            key="query_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        process_query = st.button("🚀 Process Query", type="primary")
    
    if process_query and query:
        st.markdown("---")
        
        # Process both systems
        with st.spinner("Processing with both RAG systems..."):
            traditional_result = traditional_rag_query_enhanced(
                st.session_state.llm, 
                st.session_state.vector_db, 
                query
            )
            
            agentic_result = agentic_rag_query_enhanced(
                st.session_state.llm,
                st.session_state.vector_db,
                query
            )
        
        # Display results in two columns
        col1, col2 = st.columns(2)
        
        # Traditional RAG Results
        with col1:
            st.markdown('<div class="pipeline-box traditional-rag">', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">📚 Traditional RAG</div>', unsafe_allow_html=True)
            
            # Route indicator
            st.markdown(f'<div class="routing-decision fixed-route">{traditional_result["route_decision"]}</div>', unsafe_allow_html=True)
            st.info(traditional_result["routing_explanation"])
            
            # Processing steps
            st.markdown("**Processing Pipeline:**")
            steps_html = create_processing_steps_visual(traditional_result["processing_steps"], "traditional")
            st.markdown(steps_html, unsafe_allow_html=True)
            
            # Answer
            st.markdown("**Answer:**")
            st.write(traditional_result["answer"])
            
            # Traditional RAG transparency (simpler)
            st.markdown("---")
            st.markdown("**📊 Basic Information**")
            
            with st.expander("🔄 Fixed Processing Details"):
                st.info("Traditional RAG uses a fixed, simple approach:")
                st.markdown("- **Routing:** Always LOCAL (no intelligence)")
                st.markdown("- **Documents:** Uses 2 documents maximum")
                st.markdown("- **Content:** Truncated to 300 characters per document")
                st.markdown("- **Processing:** Basic similarity matching only")
                st.markdown("- **Analysis:** No routing confidence or reasoning")
                st.warning("Limited source attribution available in Traditional RAG")
            
            # Metrics
            col1a, col1b, col1c = st.columns(3)
            with col1a:
                st.metric("Time", f"{traditional_result['processing_time']:.2f}s")
            with col1b:
                st.metric("Context", f"{traditional_result['context_length']} chars")
            with col1c:
                st.metric("Intelligence", traditional_result["intelligence_level"])
            
            # Pipeline visualization
            trad_fig = create_pipeline_visualization("traditional")
            st.plotly_chart(trad_fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Agentic RAG Results
        with col2:
            st.markdown('<div class="pipeline-box agentic-rag">', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">🤖 Agentic RAG</div>', unsafe_allow_html=True)
            
            # Route indicator
            if agentic_result["route_decision"] == "LOCAL":
                route_class = "local-route"
            elif agentic_result["route_decision"] == "WEB":
                route_class = "web-route"
            else:  # HYBRID
                route_class = "hybrid-route"
            st.markdown(f'<div class="routing-decision {route_class}">{agentic_result["route_decision"]} Route</div>', unsafe_allow_html=True)
            st.success(agentic_result["routing_explanation"])
            
            # Processing steps
            st.markdown("**Processing Pipeline:**")
            steps_html = create_processing_steps_visual(agentic_result["processing_steps"], "agentic")
            st.markdown(steps_html, unsafe_allow_html=True)
            
            # Answer
            st.markdown("**Answer:**")
            st.write(agentic_result["answer"])
            
            # ===== ENHANCED TRANSPARENCY SECTION =====
            st.markdown("---")
            st.markdown("**🔍 Transparency & Source Attribution**")
            
            # Routing Decision Details
            with st.expander("🧭 Detailed Routing Analysis", expanded=True):
                st.markdown(f"**Route Decision:** {agentic_result['route_decision']}")
                st.markdown(f"**Confidence Level:** {agentic_result.get('routing_confidence', 'Unknown').title()}")
                st.markdown(f"**Context Match Score:** {agentic_result.get('context_match_score', 0):.3f}")
                st.markdown(f"**Temporal Requirement:** {agentic_result.get('temporal_requirement', 'Unknown')}")
                st.markdown(f"**Reasoning:** {agentic_result['routing_explanation']}")
                
                # Full routing analysis
                with st.expander("📋 Complete Router Analysis"):
                    st.text(agentic_result.get('full_routing_analysis', 'Analysis not available'))
            
            # Source Details
            if agentic_result['route_decision'] in ['LOCAL', 'HYBRID']:
                # Local source details
                with st.expander("📚 Local Document Sources", expanded=True):
                    local_details = agentic_result.get('local_source_details', [])
                    if local_details:
                        st.markdown(f"**Total Chunks Used:** {len(local_details)}")
                        
                        for detail in local_details:
                            st.markdown(f"**Chunk {detail['chunk_id']}:**")
                            col_a, col_b, col_c = st.columns([2, 1, 1])
                            
                            with col_a:
                                st.markdown(f"**Source:** {detail['source_file']}")
                                st.markdown(f"**Page:** {detail.get('page', 'N/A')}")
                            with col_b:
                                st.metric("Similarity", f"{detail['similarity_score']:.3f}")
                            with col_c:
                                st.metric("Length", f"{detail['content_length']} chars")
                            
                            with st.expander(f"Preview Chunk {detail['chunk_id']}"):
                                st.text(detail['content_preview'])
                            
                            st.markdown("---")
                    else:
                        st.info("No detailed source information available")
            
            
            if agentic_result['route_decision'] in ['WEB', 'HYBRID']:
                # Web source details
                with st.expander("🌐 Web Search Sources", expanded=True):
                    web_sources = agentic_result.get('sources', [])
                    web_metadata = agentic_result.get('web_metadata', {})
                    
                    # Search metadata
                    st.markdown("**Search Information:**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Search Time", f"{web_metadata.get('search_time', 0):.3f}s")
                    with col_b:
                        st.metric("Results Found", web_metadata.get('total_results', 'Unknown'))
                    with col_c:
                        st.metric("Sources Used", len(web_sources))
                    
                    st.markdown(f"**Search Query:** {web_metadata.get('search_query', query)}")
                    
                    # Individual sources
                    st.markdown("**Source Details:**")
                    for i, source in enumerate(web_sources, 1):
                        st.markdown(f"**Source {i}: {source.get('title', 'Unknown')}**")
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.markdown(f"**Domain:** {source.get('domain', 'Unknown')}")
                            if source.get('link'):
                                st.markdown(f"**URL:** [Link]({source['link']})")
                        with col_b:
                            st.metric("Position", source.get('position', i))
                            st.metric("Snippet Length", f"{source.get('snippet_length', 0)} chars")
                        
                        with st.expander(f"Preview Source {i}"):
                            st.text(source.get('snippet', 'No preview available'))
                        
                        st.markdown("---")
            
            # Metrics
            col2a, col2b, col2c = st.columns(3)
            with col2a:
                st.metric("Time", f"{agentic_result['processing_time']:.2f}s")
            with col2b:
                st.metric("Context", f"{agentic_result['context_length']} chars")
            with col2c:
                st.metric("Intelligence", agentic_result["intelligence_level"])
            
            # Pipeline visualization
            agentic_fig = create_pipeline_visualization("agentic", agentic_result["route_decision"], agentic_result["processing_steps"])
            st.plotly_chart(agentic_fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add visual decision tree if it's an agentic query
        if agentic_result['route_decision']:
            st.markdown("---")
            st.markdown("### 🌳 Decision Tree Visualization")
            
            # Create decision tree visualization
            decision_tree_data = {
                'Query': [query[:50] + "..." if len(query) > 50 else query],
                'Router Analysis': [f"Context Match: {agentic_result.get('context_match_score', 0):.3f}"],
                'Decision': [agentic_result['route_decision']],
                'Confidence': [agentic_result.get('routing_confidence', 'unknown').title()],
                'Result': [f"{agentic_result.get('total_sources_used', 0)} sources used"]
            }
            
            tree_df = pd.DataFrame(decision_tree_data)
            st.dataframe(tree_df, use_container_width=True)
        
        # Comparison Analysis
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Detailed Comparison Analysis</div>', unsafe_allow_html=True)
        
        # Performance comparison
        comparison_data = pd.DataFrame({
            'System': ['Traditional RAG', 'Agentic RAG'],
            'Processing Time (s)': [traditional_result['processing_time'], agentic_result['processing_time']],
            'Source Type': [traditional_result['source_type'].title(), agentic_result['source_type'].title()],
            'Context Length': [traditional_result['context_length'], agentic_result['context_length']],
            'Intelligence Level': [traditional_result['intelligence_level'], agentic_result['intelligence_level']]
        })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Processing time comparison
            fig_time = px.bar(
                comparison_data, 
                x='System', 
                y='Processing Time (s)',
                title='Processing Time Comparison',
                color='System',
                color_discrete_map={
                    'Traditional RAG': '#ff6b6b',
                    'Agentic RAG': '#4ecdc4'
                }
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Context length comparison
            fig_context = px.bar(
                comparison_data,
                x='System',
                y='Context Length',
                title='Context Utilization',
                color='System',
                color_discrete_map={
                    'Traditional RAG': '#ff6b6b',
                    'Agentic RAG': '#4ecdc4'
                }
            )
            st.plotly_chart(fig_context, use_container_width=True)
        
        with col3:
            # Intelligence comparison
            intelligence_scores = {'Basic': 1, 'Advanced': 3}
            comparison_data['Intelligence Score'] = comparison_data['Intelligence Level'].map(intelligence_scores)
            
            fig_intel = px.bar(
                comparison_data,
                x='System',
                y='Intelligence Score',
                title='Intelligence Level',
                color='System',
                color_discrete_map={
                    'Traditional RAG': '#ff6b6b',
                    'Agentic RAG': '#4ecdc4'
                }
            )
            st.plotly_chart(fig_intel, use_container_width=True)
        
        # Key insights
        st.markdown("### 🎯 Key Insights & Transparency Comparison")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("""
            **📚 Traditional RAG Characteristics:**
            - 🔴 **Fixed Behavior**: Always searches local documents
            - 🔴 **Basic Retrieval**: Simple similarity matching
            - 🔴 **Limited Context**: Uses fewer documents  
            - 🔴 **No Adaptation**: Can't handle diverse query types
            - 🔴 **Poor Transparency**: Limited source attribution
            - ✅ **Predictable**: Consistent behavior and timing
            """)
        
        with insight_col2:
            st.markdown("""
            **🤖 Agentic RAG Advantages:**
            - 🟢 **Intelligent Routing**: Chooses optimal information source
            - 🟢 **Advanced Processing**: Sophisticated retrieval and analysis
            - 🟢 **Rich Context**: Uses more comprehensive information
            - 🟢 **Adaptive**: Handles diverse query types effectively
            - 🟢 **Full Transparency**: Detailed source attribution
            - 🟢 **Future-Proof**: Scales with information needs
            """)
        
        # Enhanced Transparency Summary
        st.markdown("---")
        st.markdown("### 🔍 Enhanced Transparency Features in Agentic RAG")
        
        transparency_col1, transparency_col2 = st.columns(2)
        
        with transparency_col1:
            st.markdown("""
            **🧭 Routing Intelligence:**
            - Detailed routing decision reasoning
            - Confidence scores for each decision
            - Context match scoring (0.0-1.0)
            - Temporal requirement analysis
            - Complete router analysis log
            """)
        
        with transparency_col2:
            st.markdown("""
            **📊 Source Attribution:**
            - Document chunks with similarity scores
            - Source file and page information
            - Content previews for each chunk
            - Web source rankings and metadata
            - Search query optimization details
            """)
        
        # Add to query history
        st.session_state.query_history.append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'query': query,
            'traditional_route': traditional_result['route_decision'],
            'agentic_route': agentic_result['route_decision'],
            'traditional_time': traditional_result['processing_time'],
            'agentic_time': agentic_result['processing_time'],
            'knowledge_base': st.session_state.current_knowledge_base
        })
    
    # Query History
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown('<div class="sub-header">📈 Query History & Analytics</div>', unsafe_allow_html=True)
        
        history_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Analytics
        if len(st.session_state.query_history) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Average processing times
                avg_traditional = history_df['traditional_time'].mean()
                avg_agentic = history_df['agentic_time'].mean()
                
                st.metric("Avg Traditional RAG Time", f"{avg_traditional:.2f}s")
                st.metric("Avg Agentic RAG Time", f"{avg_agentic:.2f}s")
                
            with col2:
                # Routing distribution for Agentic RAG
                route_counts = history_df['agentic_route'].value_counts()
                fig_routes = px.pie(
                    values=route_counts.values,
                    names=route_counts.index,
                    title="Agentic RAG Routing Distribution"
                )
                st.plotly_chart(fig_routes, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()

if __name__ == "__main__":
    main()