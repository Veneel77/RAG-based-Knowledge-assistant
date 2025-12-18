"""
Document loader
Extracts text from various file formats (PDF, TXT, DOCX)
"""
import os
from pathlib import Path
from typing import Optional
import PyPDF2
import docx
from backend.app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentLoader:
    """Load and extract text from various document formats"""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Extracted text content
        """
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            logger.info(f"Successfully loaded PDF: {file_path} ({len(text)} characters)")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise Exception(f"Failed to load PDF: {str(e)}")
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """
        Extract text from TXT file
        
        Args:
            file_path: Path to TXT file
        
        Returns:
            Text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            logger.info(f"Successfully loaded TXT: {file_path} ({len(text)} characters)")
            return text.strip()
        
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                logger.info(f"Successfully loaded TXT with latin-1 encoding: {file_path}")
                return text.strip()
            except Exception as e:
                logger.error(f"Error loading TXT {file_path}: {str(e)}")
                raise Exception(f"Failed to load TXT: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {str(e)}")
            raise Exception(f"Failed to load TXT: {str(e)}")
    
    @staticmethod
    def load_docx(file_path: str) -> str:
        """
        Extract text from DOCX file
        
        Args:
            file_path: Path to DOCX file
        
        Returns:
            Extracted text content
        """
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            logger.info(f"Successfully loaded DOCX: {file_path} ({len(text)} characters)")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {str(e)}")
            raise Exception(f"Failed to load DOCX: {str(e)}")
    
    @staticmethod
    def load_document(file_path: str) -> str:
        """
        Load document based on file extension
        
        Args:
            file_path: Path to document file
        
        Returns:
            Extracted text content
        
        Raises:
            ValueError: If file type is not supported
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return DocumentLoader.load_pdf(file_path)
        elif file_ext == '.txt':
            return DocumentLoader.load_txt(file_path)
        elif file_ext == '.docx':
            return DocumentLoader.load_docx(file_path)
        else:
            logger.error(f"Unsupported file type: {file_ext}")
            raise ValueError(f"Unsupported file type: {file_ext}")

