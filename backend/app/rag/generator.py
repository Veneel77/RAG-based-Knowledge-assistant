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
        Build enhanced prompt for Gemini API with deep analysis capabilities
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Optional conversation history
        
        Returns:
            Formatted prompt
        """
        prompt_parts = [
            "You are an EXPERT AI research assistant with deep analytical capabilities.",
            "Your role is to provide comprehensive, insightful, and intellectually rigorous answers.",
            "",
            "CORE CAPABILITIES:",
            "1. Deep Analysis: Extract key insights, patterns, and relationships from the context",
            "2. Critical Thinking: Identify implications, limitations, and future directions",
            "3. Synthesis: Connect ideas across different parts of the document",
            "4. Expertise: Explain complex concepts clearly while maintaining depth",
            "5. Practical Application: Suggest real-world applications and use cases",
            "",
            "ANSWER GUIDELINES:",
            "✓ Provide COMPREHENSIVE answers with multiple perspectives",
            "✓ Include KEY INSIGHTS beyond just facts - explain WHY and HOW",
            "✓ Cite specific sources [Source X] to support each claim",
            "✓ Identify IMPORTANT PATTERNS, TRENDS, or RELATIONSHIPS in the data",
            "✓ Suggest PRACTICAL APPLICATIONS or next steps when relevant",
            "✓ Point out LIMITATIONS or gaps in the provided information",
            "✓ Be INTELLECTUALLY CURIOUS - go beyond surface-level answers",
            "✓ Use STRUCTURED formatting (bullet points, sections) for clarity",
            "",
            "AVOID:",
            "✗ Generic or superficial answers",
            "✗ Ignoring important details in the context",
            "✗ Making claims without citing sources",
            "✗ Being overly brief when depth is needed",
            "",
        ]
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("=== CONVERSATION HISTORY ===")
            for entry in conversation_history[-3:]:  # Last 3 exchanges
                prompt_parts.append(f"User: {entry.get('query', '')}")
                prompt_parts.append(f"Assistant: {entry.get('response', '')[:200]}...")
            prompt_parts.append("")
        
        # Add context with emphasis on depth
        if context:
            prompt_parts.extend([
                "=== RETRIEVED CONTEXT FROM DOCUMENTS ===",
                "",
                context,
                "",
                "=== END OF CONTEXT ===",
                ""
            ])
        else:
            prompt_parts.extend([
                "⚠️ NOTE: No relevant context was found in the uploaded documents.",
                "Provide a thoughtful response based on general knowledge, but mention this limitation.",
                ""
            ])
        
        # Add current query with enhanced instructions
        prompt_parts.extend([
            "=== USER QUESTION ===",
            f"{query}",
            "",
            "=== YOUR TASK ===",
            "Provide a COMPREHENSIVE, INSIGHTFUL answer that:",
            "1. Directly answers the question with depth and clarity",
            "2. Extracts and explains KEY INSIGHTS from the context",
            "3. Identifies PATTERNS, RELATIONSHIPS, or TRENDS",
            "4. Discusses IMPLICATIONS and PRACTICAL APPLICATIONS",
            "5. Cites sources [Source X] for each major point",
            "6. Suggests FURTHER CONSIDERATIONS or next steps",
            "7. Points out any LIMITATIONS in the available information",
            "",
            "Remember: Go DEEP, not just surface-level. The user wants REAL INSIGHTS and VALUE.",
            "",
            "BEGIN YOUR COMPREHENSIVE ANSWER:"
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

