"""
Response generation module for Sapphire RAG pipeline.
Handles LLM response generation using Groq Cloud API.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    A class to generate responses using Groq LLM with retrieved context.
    """
    
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
        self.chain = self._create_chain()
    
    def _initialize_llm(self):
        """Initialize the Groq LLM."""
        try:
            logger.info(f"Initializing Groq LLM: {self.model_name}")
            llm = ChatGroq(
                temperature=0.1,
                groq_api_key=self.groq_api_key,
                model_name=self.model_name
            )
            logger.info("Groq LLM initialized successfully.")
            return llm
        except Exception as e:
            logger.error(f"Error initializing Groq LLM: {e}")
            raise
    
    def _create_prompt_template(self):
        """Create a full multi-turn prompt using context + chat history."""
        system_template = """You are a friendly, contextual, and professional customer service chatbot for Sapphire, a Pakistani fashion brand.

Your role is to help customers with:
- Product information and recommendations
- Pricing and availability
- Sizing and fabric details
- Return and exchange policies
- Shipping information
- General brand inquiries

Guidelines:
- Use the provided context to answer questions accurately
- If you don't know something, say so politely
- Keep responses concise but informative
- Maintain a helpful and professional tone
- Focus on Sapphire's products and services
- Use Pakistani Rupees (PKR) for pricing

Context from knowledge base:
{context}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        return prompt

    def _create_chain(self):
        """Create the LLM chain for response generation."""
        chain = self.prompt_template | self.llm | StrOutputParser()
        return chain
    
    def generate_response(
        self, 
        question: str, 
        context: str, 
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            question: User's question
            context: Retrieved context from vector store
            chat_history: Previous conversation history
            
        Returns:
            Generated response
        """
        try:
            # Format chat history
            formatted_history = []
            if chat_history:
                for msg in chat_history[-5:]:  # Keep last 5 messages for context
                    if msg["role"] == "user":
                        formatted_history.append(HumanMessage(content=msg["content"]))
                    else:
                        formatted_history.append(AIMessage(content=msg["content"]))
            
            # Generate response
            response = self.chain.invoke({
                "question": question,
                "context": context,
                "chat_history": formatted_history
            })
            
            logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."
    
    def generate_response_with_sources(
        self, 
        question: str, 
        retrieved_docs: List[Dict[str, Any]], 
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response with source information.
        
        Args:
            question: User's question
            retrieved_docs: List of retrieved documents with metadata
            chat_history: Previous conversation history
            
        Returns:
            Dictionary with response and sources
        """
        try:
            # Combine context from retrieved documents
            context_parts = []
            sources = []
            
            for doc in retrieved_docs:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                context_parts.append(content)
                
                # Extract source information
                source_info = {
                    'chunk_type': metadata.get('chunk_type', 'unknown'),
                    'product_id': metadata.get('product_id', ''),
                    'title': metadata.get('title', ''),
                    'category': metadata.get('category', ''),
                    'url': metadata.get('url', '')
                }
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            # Generate response
            response = self.generate_response(question, context, chat_history)
            
            return {
                'response': response,
                'sources': sources,
                'context_used': len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Error generating response with sources: {e}")
            return {
                'response': "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                'sources': [],
                'context_used': 0
            }


class ConversationManager:
    """
    Manages conversation state and history.
    """
    
    def __init__(self, max_history_length: int = 10):
        self.max_history_length = max_history_length
        self.conversations = {}  # session_id -> conversation history
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to conversation history."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "content": content
        })
        
        # Keep only recent messages
        if len(self.conversations[session_id]) > self.max_history_length:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history_length:]
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        return self.conversations.get(session_id, [])
    
    def clear_history(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]


if __name__ == "__main__":
    # Example usage
    import os
    
    # This would typically come from environment variables or config
    groq_api_key = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
    
    if groq_api_key == "your-groq-api-key-here":
        logger.warning("Please set GROQ_API_KEY environment variable for testing.")
    else:
        try:
            # Initialize response generator
            generator = ResponseGenerator(groq_api_key)
            
            # Example context
            context = """
            Product: Printed Dobby Shirt
            Price: Rs.2,990.00
            Category: Ready to Wear
            Fabric: Dobby
            Available Colors: Pink, Multi
            Available Sizes: XS, S, M, L, XL
            Care Instructions: Machine wash cold, do not bleach, tumble dry low
            """
            
            # Example conversation
            chat_history = [
                {"role": "user", "content": "What shirts do you have?"},
                {"role": "assistant", "content": "We have various shirts in our Ready to Wear collection."}
            ]
            
            question = "Tell me about the Printed Dobby Shirt"
            
            response = generator.generate_response(question, context, chat_history)
            print(f"Question: {question}")
            print(f"Response: {response}")
            
        except Exception as e:
            logger.error(f"Error in example usage: {e}")
            print(f"Error: {e}")