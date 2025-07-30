"""
Pydantic models for Sapphire RAG API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Model for a single chat message."""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the message")


class ChatRequest(BaseModel):
    """Model for chat request."""
    message: str = Field(..., description="User's message/question")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation tracking")
    chat_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation history")
    include_sources: Optional[bool] = Field(default=False, description="Whether to include source information")


class SourceInfo(BaseModel):
    """Model for source information."""
    chunk_type: str = Field(..., description="Type of chunk (product, faq, policy, etc.)")
    product_id: Optional[str] = Field(default="", description="Product ID if applicable")
    title: Optional[str] = Field(default="", description="Title of the source")
    category: Optional[str] = Field(default="", description="Category of the source")
    url: Optional[str] = Field(default="", description="URL of the source if available")


class ChatResponse(BaseModel):
    """Model for chat response."""
    response: str = Field(..., description="Generated response")
    session_id: str = Field(..., description="Session ID")
    sources: Optional[List[SourceInfo]] = Field(default=[], description="Source information")
    context_used: Optional[int] = Field(default=0, description="Number of context chunks used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(default="1.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Model for error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class SearchRequest(BaseModel):
    """Model for search request."""
    query: str = Field(..., description="Search query")
    limit: Optional[int] = Field(default=5, description="Maximum number of results")
    filter_type: Optional[str] = Field(default=None, description="Filter by chunk type")


class SearchResult(BaseModel):
    """Model for a single search result."""
    content: str = Field(..., description="Content of the result")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Metadata of the result")


class SearchResponse(BaseModel):
    """Model for search response."""
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original query")
    timestamp: datetime = Field(default_factory=datetime.now, description="Search timestamp")


class ConfigRequest(BaseModel):
    """Model for configuration update request."""
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    model_name: Optional[str] = Field(default=None, description="LLM model name")
    temperature: Optional[float] = Field(default=None, description="LLM temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for response")


class ConfigResponse(BaseModel):
    """Model for configuration response."""
    message: str = Field(..., description="Configuration update message")
    current_config: Dict[str, Any] = Field(..., description="Current configuration")
    timestamp: datetime = Field(default_factory=datetime.now, description="Configuration timestamp")


class StatsResponse(BaseModel):
    """Model for statistics response."""
    total_chunks: int = Field(..., description="Total number of chunks in vector store")
    total_conversations: int = Field(..., description="Total number of active conversations")
    uptime: str = Field(..., description="API uptime")
    timestamp: datetime = Field(default_factory=datetime.now, description="Statistics timestamp")


# Request/Response models for batch operations
class BatchChatRequest(BaseModel):
    """Model for batch chat request."""
    messages: List[str] = Field(..., description="List of messages to process")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation tracking")
    include_sources: Optional[bool] = Field(default=False, description="Whether to include source information")


class BatchChatResponse(BaseModel):
    """Model for batch chat response."""
    responses: List[ChatResponse] = Field(..., description="List of responses")
    session_id: str = Field(..., description="Session ID")
    total_processed: int = Field(..., description="Total number of messages processed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch response timestamp")


# Models for data management
class ChunkInfo(BaseModel):
    """Model for chunk information."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_type: str = Field(..., description="Type of chunk")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")


class DataUploadRequest(BaseModel):
    """Model for data upload request."""
    data_type: str = Field(..., description="Type of data (product, faq, policy, etc.)")
    data: List[Dict[str, Any]] = Field(..., description="Data to upload")
    overwrite: Optional[bool] = Field(default=False, description="Whether to overwrite existing data")


class DataUploadResponse(BaseModel):
    """Model for data upload response."""
    message: str = Field(..., description="Upload status message")
    chunks_created: int = Field(..., description="Number of chunks created")
    chunks_updated: int = Field(..., description="Number of chunks updated")
    timestamp: datetime = Field(default_factory=datetime.now, description="Upload timestamp")


# Models for analytics
class ConversationAnalytics(BaseModel):
    """Model for conversation analytics."""
    session_id: str = Field(..., description="Session ID")
    total_messages: int = Field(..., description="Total messages in conversation")
    start_time: datetime = Field(..., description="Conversation start time")
    last_activity: datetime = Field(..., description="Last activity time")
    topics: List[str] = Field(default=[], description="Topics discussed")


class AnalyticsResponse(BaseModel):
    """Model for analytics response."""
    total_conversations: int = Field(..., description="Total number of conversations")
    active_conversations: int = Field(..., description="Number of active conversations")
    popular_topics: List[str] = Field(..., description="Most popular topics")
    average_conversation_length: float = Field(..., description="Average conversation length")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analytics timestamp")

