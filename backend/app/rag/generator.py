"""
Generator using Google Gemini API
Generates responses based on retrieved context
"""
import os
from typing import Optional, List, Dict
import google.generativeai as genai
from backend.app.config import get_settings
from backend.app.utils.logger import setup_logger

settings = get_settings()
logger = setup_logger(__name__)


class Generator:
    """Generate responses using Gemini API"""
    
    def __init__(self):
        """Initialize Gemini API"""
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set in environment")
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in your .env file or environment variables."
            )
        
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info(f"Initialized Gemini API with model: {settings.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise
    
    def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate response based on query and context
        
        Args:
            query: User query
            context: Retrieved context from documents
            conversation_history: Optional conversation history
        
        Returns:
            Generated response
        """
        try:
            # Build prompt
            prompt = self._build_prompt(query, context, conversation_history)
            
            logger.info(f"Generating response for query: {query[:100]}...")
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                logger.warning("Empty response from Gemini API")
                return "I apologize, but I couldn't generate a response. Please try again."
            
            logger.info("Successfully generated response")
            return response.text.strip()
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build prompt for Gemini API
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Optional conversation history
        
        Returns:
            Formatted prompt
        """
        prompt_parts = [
            "You are a helpful AI assistant that answers questions based on the provided context.",
            "Your task is to provide accurate, concise, and helpful answers based solely on the information given.",
            "",
            "Guidelines:",
            "- Answer based ONLY on the provided context",
            "- If the context doesn't contain enough information, say so honestly",
            "- Cite sources when possible by referencing [Source X]",
            "- Be concise but thorough",
            "- Use a professional and friendly tone",
            "",
        ]
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for entry in conversation_history[-3:]:  # Last 3 exchanges
                prompt_parts.append(f"User: {entry.get('query', '')}")
                prompt_parts.append(f"Assistant: {entry.get('response', '')}")
            prompt_parts.append("")
        
        # Add context
        if context:
            prompt_parts.extend([
                "Context from documents:",
                "---",
                context,
                "---",
                ""
            ])
        else:
            prompt_parts.extend([
                "Note: No relevant context was found in the documents.",
                ""
            ])
        
        # Add current query
        prompt_parts.extend([
            f"User Question: {query}",
            "",
            "Please provide a helpful answer based on the context above:"
        ])
        
        return "\n".join(prompt_parts)
    
    def check_api_key(self) -> bool:
        """
        Check if API key is valid
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Simple test to verify API key
            test_model = genai.GenerativeModel(settings.GEMINI_MODEL)
            test_model.generate_content("Hello")
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False

