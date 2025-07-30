"""
FastAPI routes for Sapphire RAG API.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .models import (
    ChatRequest, ChatResponse, HealthResponse, ErrorResponse,
    SearchRequest, SearchResponse, SearchResult,
    ConfigRequest, ConfigResponse, StatsResponse,
    BatchChatRequest, BatchChatResponse,
    SourceInfo, ChatMessage
)
from ..embeddings.vector_store import LangChainVectorStore, SentenceTransformerEmbeddings
from ..generation.response_generator import ResponseGenerator, ConversationManager
from ..retrieval.retriever import AdvancedRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for application state
app_state = {
    "vector_store": None,
    "response_generator": None,
    "conversation_manager": None,
    "retriever": None,
    "start_time": datetime.now(),
    "config": {
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "model_name": os.getenv("MODEL_NAME", "mixtral-8x7b-32768"),
        "vector_store_path": os.getenv("VECTOR_STORE_PATH", "./data/vector_store_langchain"),
        "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "1000"))
    }
}

def get_vector_store():
    """Dependency to get vector store."""
    if app_state["vector_store"] is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    return app_state["vector_store"]

def get_response_generator():
    """Dependency to get response generator."""
    if app_state["response_generator"] is None:
        raise HTTPException(status_code=500, detail="Response generator not initialized")
    return app_state["response_generator"]

def get_conversation_manager():
    """Dependency to get conversation manager."""
    if app_state["conversation_manager"] is None:
        raise HTTPException(status_code=500, detail="Conversation manager not initialized")
    return app_state["conversation_manager"]

def get_retriever():
    """Dependency to get retriever."""
    if app_state["retriever"] is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    return app_state["retriever"]

def initialize_app():
    """Initialize the application components."""
    try:
        logger.info("Initializing Sapphire RAG API...")
        
        # Load config from config.yaml environment section
        import yaml
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
            env = config.get("environment", {})
            model_llm = config.get("model", {}).get("llm", {})
            app_state["config"]["groq_api_key"] = env.get("groq_api_key", os.getenv("GROQ_API_KEY", ""))
            app_state["config"]["model_name"] = model_llm.get("model_name", os.getenv("MODEL_NAME", "mixtral-8x7b-32768"))
            app_state["config"]["temperature"] = model_llm.get("temperature", 0.1)
            app_state["config"]["max_tokens"] = model_llm.get("max_tokens", 1000)
            app_state["config"]["vector_store_path"] = env.get("vector_store_path", "./data/vector_store_langchain")
        
        # Check if Groq API key is provided
        if not app_state["config"]["groq_api_key"]:
            logger.warning("GROQ_API_KEY not provided. Some features may not work.")
        
        # Initialize vector store
        logger.info("Loading vector store...")
        vector_store = LangChainVectorStore(store_type="faiss")
        vector_store.load(app_state["config"]["vector_store_path"])
        app_state["vector_store"] = vector_store
        
        # Initialize response generator
        if app_state["config"]["groq_api_key"]:
            logger.info("Initializing response generator...")
            response_generator = ResponseGenerator(
                groq_api_key=app_state["config"]["groq_api_key"],
                model_name=app_state["config"]["model_name"]
            )
            app_state["response_generator"] = response_generator
        
        # Initialize conversation manager
        logger.info("Initializing conversation manager...")
        conversation_manager = ConversationManager()
        app_state["conversation_manager"] = conversation_manager
        
        # Initialize retriever
        if app_state["config"]["groq_api_key"]:
            logger.info("Initializing advanced retriever...")
            retriever = AdvancedRetriever(
                vector_store_path=app_state["config"]["vector_store_path"],
                llm_model_name=app_state["config"]["model_name"],
                groq_api_key=app_state["config"]["groq_api_key"]
            )
            app_state["retriever"] = retriever
        
        logger.info("Sapphire RAG API initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        raise

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="Sapphire RAG API",
        description="RAG-powered chatbot API for Sapphire fashion brand",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize application on startup."""
        initialize_app()
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.0.0"
        )
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest,
        conversation_manager: ConversationManager = Depends(get_conversation_manager),
        vector_store: LangChainVectorStore = Depends(get_vector_store)
    ):
        """Main chat endpoint."""
        try:
            # Generate session ID if not provided
            session_id = request.session_id or str(uuid.uuid4())
            
            # Get conversation history
            chat_history = conversation_manager.get_history(session_id)
            
            # Add current message to history
            conversation_manager.add_message(session_id, "user", request.message)
            
            # Search for relevant documents
            search_results = vector_store.search(request.message, k=5)
            
            # Convert search results to the expected format
            retrieved_docs = []
            for doc in search_results:
                retrieved_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            # Check if response generator is available
            if app_state["response_generator"] is None:
                # Fallback response when Groq API key is not available
                if retrieved_docs:
                    # Return the most relevant document content as response
                    response_text = f"Based on our knowledge base: {retrieved_docs[0]['content'][:500]}..."
                else:
                    response_text = "I found some information but cannot generate a proper response without the API key. Please contact customer support."
                
                sources = []
                context_used = len(retrieved_docs)
            else:
                # Generate response using LLM
                response_generator = app_state["response_generator"]
                
                if request.include_sources:
                    result = response_generator.generate_response_with_sources(
                        question=request.message,
                        retrieved_docs=retrieved_docs,
                        chat_history=chat_history
                    )
                    
                    # Convert sources to SourceInfo objects
                    sources = [SourceInfo(**source) for source in result['sources']]
                    
                    response_text = result['response']
                    context_used = result['context_used']
                else:
                    # Combine context from retrieved documents
                    context = "\n\n".join([doc['content'] for doc in retrieved_docs])
                    
                    response_text = response_generator.generate_response(
                        question=request.message,
                        context=context,
                        chat_history=chat_history
                    )
                    sources = []
                    context_used = len(retrieved_docs)
            
            # Add response to conversation history
            conversation_manager.add_message(session_id, "assistant", response_text)
            
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                sources=sources,
                context_used=context_used
            )
            
        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}")
            # Return a user-friendly error message instead of raising HTTPException
            error_response = "I apologize, but I'm experiencing technical difficulties. Please try again later."
            return ChatResponse(
                response=error_response,
                session_id=request.session_id or str(uuid.uuid4()),
                sources=[],
                context_used=0
            )
    
    @app.post("/chat/advanced", response_model=ChatResponse)
    async def chat_advanced(
        request: ChatRequest,
        retriever: AdvancedRetriever = Depends(get_retriever),
        conversation_manager: ConversationManager = Depends(get_conversation_manager)
    ):
        """Advanced chat endpoint using the full RAG pipeline."""
        try:
            # Generate session ID if not provided
            session_id = request.session_id or str(uuid.uuid4())
            
            # Get conversation history
            chat_history = conversation_manager.get_history(session_id)
            
            # Add current message to history
            conversation_manager.add_message(session_id, "user", request.message)
            
            # Get response using advanced retriever
            response_text = retriever.get_response(request.message, chat_history)
            
            # Add response to conversation history
            conversation_manager.add_message(session_id, "assistant", response_text)
            
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                sources=[],  # Advanced retriever doesn't return sources separately
                context_used=0
            )
            
        except Exception as e:
            logger.error(f"Error in advanced chat endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/search", response_model=SearchResponse)
    async def search(
        request: SearchRequest,
        vector_store: LangChainVectorStore = Depends(get_vector_store)
    ):
        """Search endpoint for finding relevant documents."""
        try:
            # Perform search
            search_results = vector_store.search(request.query, k=request.limit)
            
            # Convert results
            results = []
            for doc in search_results:
                results.append(SearchResult(
                    content=doc.page_content,
                    score=1.0,  # LangChain doesn't return scores by default
                    metadata=doc.metadata
                ))
            
            return SearchResponse(
                results=results,
                total_results=len(results),
                query=request.query
            )
            
        except Exception as e:
            logger.error(f"Error in search endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/conversations/{session_id}/history")
    async def get_conversation_history(
        session_id: str,
        conversation_manager: ConversationManager = Depends(get_conversation_manager)
    ):
        """Get conversation history for a session."""
        try:
            history = conversation_manager.get_history(session_id)
            return {"session_id": session_id, "history": history}
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/conversations/{session_id}")
    async def clear_conversation(
        session_id: str,
        conversation_manager: ConversationManager = Depends(get_conversation_manager)
    ):
        """Clear conversation history for a session."""
        try:
            conversation_manager.clear_history(session_id)
            return {"message": f"Conversation {session_id} cleared successfully"}
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/stats", response_model=StatsResponse)
    async def get_stats(
        conversation_manager: ConversationManager = Depends(get_conversation_manager)
    ):
        """Get API statistics."""
        try:
            uptime = datetime.now() - app_state["start_time"]
            uptime_str = str(uptime).split('.')[0]  # Remove microseconds
            
            return StatsResponse(
                total_chunks=1046,  # This should be dynamic based on vector store
                total_conversations=len(conversation_manager.conversations),
                uptime=uptime_str
            )
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/config", response_model=ConfigResponse)
    async def update_config(request: ConfigRequest):
        """Update API configuration."""
        try:
            updated_fields = []
            
            if request.groq_api_key is not None:
                app_state["config"]["groq_api_key"] = request.groq_api_key
                updated_fields.append("groq_api_key")
            
            if request.model_name is not None:
                app_state["config"]["model_name"] = request.model_name
                updated_fields.append("model_name")
            
            if request.temperature is not None:
                app_state["config"]["temperature"] = request.temperature
                updated_fields.append("temperature")
            
            if request.max_tokens is not None:
                app_state["config"]["max_tokens"] = request.max_tokens
                updated_fields.append("max_tokens")
            
            # Reinitialize components if necessary
            if "groq_api_key" in updated_fields or "model_name" in updated_fields:
                if app_state["config"]["groq_api_key"]:
                    app_state["response_generator"] = ResponseGenerator(
                        groq_api_key=app_state["config"]["groq_api_key"],
                        model_name=app_state["config"]["model_name"]
                    )
            
            return ConfigResponse(
                message=f"Configuration updated: {', '.join(updated_fields)}",
                current_config=app_state["config"]
            )
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc)
            ).dict()
        )
    
    return app

# Create the FastAPI app instance
app = create_app()